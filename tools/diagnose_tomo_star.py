#!/usr/bin/env python
"""
Decide why a tomo reconstruction came out featureless, without touching a GPU.

    python diagnose_tomo_star.py run_optimisation_set.star

Every check prints OK or a diagnosis. The checks are ordered so that the first
failure is the one to act on.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

from xmipp_metadata.metadata.relion_tomo import (
    _read_star, _optimisation_block, read_optimisation_set,
    read_tomograms_star, tomo_star_to_tilt_particles,
    relion_angles_to_matrix, _parse_vector,
)

BAD = []


def check(ok, label, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        BAD.append(label)
    return ok


def main(path):
    path = os.path.abspath(path)
    print(f"\n=== 1. what the file points at =========================\n{path}")

    blocks = _read_star(path)
    print("  blocks: " + ", ".join(
        f"{k or '<unnamed>'}({len(v)}x{len(v.columns)})" for k, v in blocks.items()))

    if _optimisation_block(blocks) is not None:
        opt = read_optimisation_set(path)
        for k, v in opt.items():
            print(f"  {k:<14} {v}")
        particles_star = opt["particles"]
        tomograms_star = opt["tomograms"]
        check(particles_star and os.path.exists(particles_star),
              "particles file exists")
        check(tomograms_star and os.path.exists(tomograms_star),
              "tomograms file exists")
    else:
        particles_star, tomograms_star = path, None
        print("  (not an optimisation set -- treating it as the particles file)")

    blocks = _read_star(particles_star)
    parts = blocks["particles"] if "particles" in blocks else \
        max(blocks.values(), key=lambda d: len(d.columns))
    optics = blocks.get("optics")
    star_dir = os.path.dirname(particles_star)

    print(f"\n=== 2. optics ==========================================")
    if optics is not None:
        for c in ("rlnImagePixelSize", "rlnImageSize", "rlnTomoTiltSeriesPixelSize",
                  "rlnCtfDataAreCtfPremultiplied", "rlnVoltage",
                  "rlnSphericalAberration", "rlnAmplitudeContrast", "rlnImageDimensionality"):
            if c in optics.columns:
                print(f"  {c:<32} {list(optics[c])}")
        apix_img = float(optics["rlnImagePixelSize"].iloc[0]) \
            if "rlnImagePixelSize" in optics.columns else None
        premult = int(optics["rlnCtfDataAreCtfPremultiplied"].iloc[0]) \
            if "rlnCtfDataAreCtfPremultiplied" in optics.columns else None
        check(apix_img is not None, "rlnImagePixelSize present",
              f"--sr must be {apix_img}" if apix_img else "")
        if premult is not None:
            check(True, "rlnCtfDataAreCtfPremultiplied",
                  f"= {premult}  ->  --ctf_type "
                  f"{'premultiplied' if premult else 'apply'}")
        else:
            print("  [warn] no rlnCtfDataAreCtfPremultiplied: images are NOT "
                  "premultiplied, use --ctf_type apply")
    else:
        apix_img = None
        print("  no optics block")

    print(f"\n=== 3. are these refined poses? ========================")
    for c in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"):
        v = parts[c].to_numpy(float) if c in parts.columns else None
        if v is None:
            check(False, f"{c} present", "an unrefined file -> featureless average")
            continue
        check(np.std(v) > 1.0, f"{c} varies",
              f"std={np.std(v):8.2f}  unique={len(np.unique(np.round(v, 3))):6d} / {len(v)}")

    if "rlnTomoSubtomogramRot" in parts.columns:
        sub = parts[["rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt",
                     "rlnTomoSubtomogramPsi"]].to_numpy(float)
        print(f"  rlnTomoSubtomogram* present, std = {np.std(sub, axis=0).round(2)}")
    else:
        print("  rlnTomoSubtomogram* absent -> A_sub = identity")

    for c in ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"):
        if c in parts.columns:
            v = np.abs(parts[c].to_numpy(float))
            box_ang = (apix_img or 1.0) * float(optics["rlnImageSize"].iloc[0]) \
                if optics is not None and "rlnImageSize" in optics.columns else np.inf
            check(v.max() < 0.5 * box_ang, f"{c} within half a box",
                  f"max={v.max():.1f} A   (half box = {0.5 * box_ang:.1f} A)")

    print(f"\n=== 4. images on disk vs visible frames ================")
    if "rlnTomoVisibleFrames" in parts.columns and "rlnImageName" in parts.columns:
        try:
            import mrcfile
        except ImportError:
            mrcfile = None
            print("  (mrcfile not installed -- skipping the slice-count check)")

        n_show = min(5, len(parts))
        mismatch = missing = 0
        for i in range(n_show):
            vis = _parse_vector(parts["rlnTomoVisibleFrames"].iloc[i])
            n_vis = int(vis.sum())
            name = str(parts["rlnImageName"].iloc[i])
            fn = name.split("@")[-1]
            for cand in (fn, os.path.join(star_dir, fn),
                         os.path.join(os.path.dirname(star_dir), fn),
                         os.path.join(star_dir, "..", "..", fn)):
                if os.path.exists(cand):
                    fn = cand
                    break
            exists = os.path.exists(fn)
            n_slices = None
            if exists and mrcfile is not None:
                with mrcfile.mmap(fn, permissive=True) as m:
                    n_slices = m.data.shape[0] if m.data.ndim == 3 else 1
            flag = "" if (n_slices is None or n_slices == n_vis) else "  <-- MISMATCH"
            if n_slices is not None and n_slices != n_vis:
                mismatch += 1
            if not exists:
                missing += 1
            print(f"  {name[:70]:<70} exists={exists}  visible={n_vis}  "
                  f"slices={n_slices}{flag}")
        check(missing == 0, "image stacks found on disk",
              "run from the RELION project root: paths are project-relative")
        if missing == 0:
            check(mismatch == 0, "stack slice count == visible-frame count",
                  "a mismatch pairs every image with the wrong pose -> featureless ball")
    else:
        print("  no rlnTomoVisibleFrames / rlnImageName -> not a 2D-stack data set")

    print(f"\n=== 5. the composed poses =============================")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = tomo_star_to_tilt_particles(path, shifts="from_origin",
                                         shift_units="pixel")
    for w in caught:
        print(f"  [warning] {str(w.message)[:160]}")

    print(f"  {len(df)} rows, {int(df['subtomo_labels'].max())} particles")

    # across particles at one frame: must be wide, or the pose is not reaching the output
    f0 = df["rlnTomoFrameIndex"].mode().iloc[0]
    one = df[df["rlnTomoFrameIndex"] == f0]
    spread = one[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].std().to_numpy()
    check(spread.max() > 10.0, "pose varies ACROSS particles at a fixed tilt",
          f"std={np.round(spread, 2)}  (small => A_part is not reaching the output)")

    # across frames of one particle: must track the tilt range
    p1 = df[df["subtomo_labels"] == 1]
    spread2 = p1[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].std().to_numpy()
    check(spread2.max() > 5.0, "pose varies ACROSS tilts of one particle",
          f"std={np.round(spread2, 2)}  (small => the tilt geometry is not composed)")

    d = df.groupby("rlnTomoFrameIndex")["rlnDefocusU"].agg(lambda s: s.max() - s.min())
    check(d.max() > 0, "per-particle defocus gradient present",
          f"max spread {d.max():.0f} A")

    sx = df["rlnOriginX"].to_numpy() if "rlnOriginX" in df.columns else np.zeros(1)
    box = float(optics["rlnImageSize"].iloc[0]) if optics is not None \
        and "rlnImageSize" in optics.columns else np.inf
    check(np.abs(sx).max() < 0.5 * box, "shifts smaller than half a box",
          f"max |shiftX| = {np.abs(sx).max():.2f} px  (half box = {0.5 * box:.1f})")

    print("\n=== verdict ============================================")
    if BAD:
        for b in BAD:
            print(f"  FAILED: {b}")
    else:
        print("  every metadata check passes -- the metadata is not the problem;\n"
              "  suspect --sr / --ctf_type / the image paths instead")
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run_optimisation_set.star")
