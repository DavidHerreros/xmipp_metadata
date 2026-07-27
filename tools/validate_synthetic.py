#!/usr/bin/env python
"""
Validate a synthetic data set built by ``make_synthetic_tomo.py``, in two stages.

    python validate_synthetic.py /path/to/synth_relion

Stage 1 reconstructs from ``ground_truth/poses.npz`` -- the poses actually used to
render the images. If that does not reproduce the phantom, the *test* is broken and
nothing it says about the converter means anything. Stage 2 reconstructs from the poses
the converter derives from the star files. Stage 2 is the real test; stage 1 is what
makes a stage 2 failure attributable.

The reconstruction here is a plain numpy Fourier-slice gridding, deliberately not hax's:
this script answers "is the metadata right", and bringing in the GPU reconstructor would
mix that question with a second one. Run hax afterwards for the end-to-end check.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import starfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_synthetic_tomo import ctf as make_ctf, dose_weight  # noqa: E402  (shared definitions, not conventions)


def _slice_grid(box):
    g = (np.fft.fftshift(np.fft.fftfreq(box)) * box).astype(np.float64)
    f0, f1 = np.meshgrid(g, g, indexing="ij")            # f0 <-> row (y), f1 <-> col (x)
    return f0, f1


def relion_matrix(rot, tilt, psi):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_euler("ZYZ", [rot, tilt, psi], degrees=True).as_matrix().T


def reconstruct(images, angles, shifts, weights, box, premultiplied=True):
    """Wiener gridding with trilinear insertion.

    The denominator is always sum((CTF*W)^2). The numerator depends on what the stack
    holds: a pre-multiplied image already carries one factor of CTF*W, a plain
    observation still needs it applied here.
    """
    f0, f1 = _slice_grid(box)
    num = np.zeros((box,) * 3, np.complex128)
    den = np.zeros((box,) * 3, np.float64)

    for img, ang, sh, w in zip(images, angles, shifts, weights):
        ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
        # undo the stored translation, same sense as the reconstructor under test:
        # the content sits at (p - shift), so divide that phase out
        ft = ft * np.exp(-2j * np.pi * (sh[1] * f0 + sh[0] * f1) / box)
        if not premultiplied:
            ft = ft * w

        R = relion_matrix(*ang)
        kx = R[0, 0] * f1 + R[1, 0] * f0
        ky = R[0, 1] * f1 + R[1, 1] * f0
        kz = R[0, 2] * f1 + R[1, 2] * f0

        for sign in (1.0, -1.0):
            v = ft if sign > 0 else np.conj(ft)
            pos = np.stack([sign * kz, sign * ky, sign * kx], 0) + box // 2
            base = np.floor(pos).astype(np.int64)
            frac = pos - base
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        idx = base + np.array([dz, dy, dx])[:, None, None]
                        wt = ((frac[0] if dz else 1 - frac[0]) *
                              (frac[1] if dy else 1 - frac[1]) *
                              (frac[2] if dx else 1 - frac[2]))
                        ok = np.all((idx >= 0) & (idx < box), axis=0)
                        z, y, x = [np.clip(idx[i], 0, box - 1) for i in range(3)]
                        np.add.at(num, (z[ok], y[ok], x[ok]), (v * wt)[ok])
                        np.add.at(den, (z[ok], y[ok], x[ok]), (w ** 2 * wt)[ok])

    ft = num / (den + 0.001 * den.mean())
    vol = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ft))))

    r = np.fft.fftshift(np.fft.fftfreq(box))
    z, y, x = np.meshgrid(r, r, r, indexing="ij")
    return vol / np.maximum((np.sinc(x) * np.sinc(y) * np.sinc(z)) ** 2, 1e-2)


def fsc(a, b):
    A, B = np.fft.fftn(a), np.fft.fftn(b)
    n = a.shape[0]
    g = np.fft.fftfreq(n) * n
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    shell = np.round(np.sqrt(x ** 2 + y ** 2 + z ** 2)).astype(int)
    nb = n // 2 + 1
    num = np.bincount(shell.ravel(), (A * np.conj(B)).real.ravel(), nb)
    d1 = np.bincount(shell.ravel(), (np.abs(A) ** 2).ravel(), nb)
    d2 = np.bincount(shell.ravel(), (np.abs(B) ** 2).ravel(), nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(num / np.sqrt(d1 * d2))


def report(name, vol, truth):
    z = lambda v: (v - v.mean()) / (v.std() or 1.0)
    cc = float((z(vol) * z(truth)).mean())
    cc_m = float((z(vol[::-1, ::-1, ::-1]) * z(truth)).mean())
    c = fsc(z(vol), z(truth))
    med = float(np.median(c[1:len(c) - 1]))
    print(f"  {name:<34} CC {cc:+.4f}   mirrored CC {cc_m:+.4f}   median FSC {med:.4f}")
    return cc, cc_m, med


def load_images(md, box):
    return np.stack([md.getMetaDataImage(i)[0] if md.getMetaDataImage(i).ndim == 3
                     else md.getMetaDataImage(i) for i in range(len(md))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--entry", default=None, help="star file (default: read from README)")
    ap.add_argument("--break", dest="sabotage", default=None,
                    choices=("transpose", "mirror", "drop-shifts", "swap-rot-psi"),
                    help="deliberately corrupt the converted poses. A test that cannot "
                         "fail proves nothing, so this is how you check it can.")
    args = ap.parse_args()

    root = os.path.abspath(args.project)
    import mrcfile
    with mrcfile.open(os.path.join(root, "ground_truth", "phantom.mrc")) as m:
        truth = np.asarray(m.data, np.float64)
    gt = np.load(os.path.join(root, "ground_truth", "poses.npz"))
    box, apix = int(gt["box"]), float(gt["apix"])

    entry = args.entry
    if entry is None:
        for line in open(os.path.join(root, "README.md")):
            if line.strip().startswith("particles      :"):
                entry = line.split(":", 1)[1].strip()
                break
    star = os.path.join(root, entry)
    print(f"project {root}\nentry   {entry}\nbox {box}  apix {apix:.4f}\n")

    cwd = os.getcwd()
    os.chdir(root)
    try:
        from xmipp_metadata.metadata import XmippMetaData
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = XmippMetaData(star)
        assert md.binaries, "the stacks were not found"
        images = load_images(md, box).astype(np.float64)

        # the weight the images were pre-multiplied by, rebuilt from the converted
        # metadata: CTF (with its per-particle depth term) times the dose weight
        cols = {c: md.getMetaDataColumns(c) for c in
                ("ctfDefocusU", "ctfDefocusV", "ctfDefocusAngle",
                 "ctfSphericalAberration", "ctfVoltage", "ctfScaleFactor", "preExposure")
                if md.isMetaDataLabel(c)}
        n = len(md)
        weights = np.stack([
            make_ctf(box, apix, cols["ctfDefocusU"][i], cols["ctfDefocusV"][i],
                     cols["ctfDefocusAngle"][i], cols["ctfVoltage"][i],
                     cols["ctfSphericalAberration"][i], 0.07,
                     scale=cols.get("ctfScaleFactor", np.ones(n))[i])
            * (dose_weight(box, apix, cols["preExposure"][i])
               if "preExposure" in cols else 1.0)
            for i in range(n)]).astype(np.float64)

        # Getting this wrong silently destroys the reconstruction, and a silent wrong
        # answer is the failure this whole exercise exists to prevent -- so read it from
        # the converted table, and fall back to scanning the file's own optics block.
        premult = True
        if md.isMetaDataLabel("rlnCtfDataAreCtfPremultiplied"):
            premult = bool(int(md.getMetaDataColumns("rlnCtfDataAreCtfPremultiplied")[0]))
        else:
            for blk in starfile.read(star, always_dict=True).values():
                labels = (blk.columns if hasattr(blk, "columns") else
                          blk.index if hasattr(blk, "index") else blk.keys())
                if "rlnCtfDataAreCtfPremultiplied" in labels:
                    premult = bool(int(np.asarray(blk["rlnCtfDataAreCtfPremultiplied"]).ravel()[0]))
                    break

        conv_ang = np.stack([md.getMetaDataColumns("angleRot"),
                             md.getMetaDataColumns("angleTilt"),
                             md.getMetaDataColumns("anglePsi")], axis=1)
        conv_sh = np.stack([md.getMetaDataColumns("shiftX"),
                            md.getMetaDataColumns("shiftY")], axis=1)
    finally:
        os.chdir(cwd)
    print(f"stored images: {'pre-multiplied' if premult else 'plain observations'}\n")

    print("stage 1 -- the renderer, using the poses the images were made with")
    gt_ang = np.stack([gt["rot"], gt["tilt"], gt["psi"]], axis=1)
    gt_sh = np.stack([gt["shift_x"], gt["shift_y"]], axis=1)
    v1 = reconstruct(images, gt_ang, gt_sh, weights, box, premult)
    r1 = report("ground-truth poses", v1, truth)

    if args.sabotage:
        print(f"\n!! sabotaging the converted poses: {args.sabotage}")
        if args.sabotage == "transpose":
            conv_ang = conv_ang[:, ::-1] * np.array([1.0, 1.0, 1.0])
        elif args.sabotage == "mirror":
            conv_ang[:, 1] = 180.0 - conv_ang[:, 1]
        elif args.sabotage == "drop-shifts":
            conv_sh = np.zeros_like(conv_sh)
        elif args.sabotage == "swap-rot-psi":
            conv_ang = conv_ang[:, [2, 1, 0]]

    print("\nstage 2 -- the converter, using the poses it derived from the star file")
    v2 = reconstruct(images, conv_ang, conv_sh, weights, box, premult)
    r2 = report("converted poses", v2, truth)

    agree = float(np.median(fsc(v1, v2)[1:box // 2]))
    print(f"\n  agreement between the two              median FSC {agree:.4f}")

    print()
    # The verdict keys on stage1-vs-stage2, not on either against the phantom. The two
    # pose sets describe the same images and must agree exactly; agreeing with the
    # phantom is bounded by interpolation, the Wiener floor and how much of Fourier
    # space this many particles actually cover, none of which is a convention question.
    if agree < 0.99:
        print(f"FAIL: the converter's poses differ from the ones the images were "
              f"rendered with (agreement {agree:.4f})")
        if r2[1] > r2[0] + 0.1:
            print("      the map is MIRRORED -- handedness / tilt sign")
    elif r1[2] < 0.8:
        print(f"INCONCLUSIVE: the converter agrees with the renderer, but the renderer "
              f"itself only reaches {r1[2]:.3f} against the phantom.")
        print("      Use more particles or more tilts before trusting this.")
    else:
        print(f"PASS: the converter's poses are the ones the images were rendered with "
              f"(agreement {agree:.4f}),")
        print(f"      and those poses reproduce the phantom (median FSC {r1[2]:.3f}).")


if __name__ == "__main__":
    main()
