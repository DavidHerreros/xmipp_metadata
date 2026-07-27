#!/usr/bin/env python
"""
Compare two reconstructions voxel-by-voxel and by FSC.

    python compare_volumes.py reference.mrc test.mrc [--apix 4.0]

Two reconstructions built from the same particles and poses must agree whatever those
poses are, so the maps do not have to be any good for the comparison to mean something.
FSC ~ 1 to Nyquist means the reconstructors agree; a dip from the lowest shells means a
convention mismatch, a dip only at high frequency a weighting difference.
"""

import argparse
import sys

import numpy as np


def read_mrc(path):
    import mrcfile
    with mrcfile.open(path, permissive=True) as m:
        vol = np.asarray(m.data, dtype=np.float32)
        try:
            apix = float(m.voxel_size.x)
        except Exception:
            apix = None
    return vol, (apix if apix and apix > 0 else None)


def fsc(a, b):
    """Shell-averaged Fourier ring correlation between two equal-sized volumes."""
    A = np.fft.fftn(a)
    B = np.fft.fftn(b)

    n = a.shape[0]
    freq = np.fft.fftfreq(n) * n
    r = np.sqrt(sum(g ** 2 for g in np.meshgrid(freq, freq, freq, indexing="ij")))
    shell = np.round(r).astype(int)

    nbins = n // 2 + 1
    num = np.bincount(shell.ravel(), (A * np.conj(B)).real.ravel(), nbins)
    d1 = np.bincount(shell.ravel(), (np.abs(A) ** 2).ravel(), nbins)
    d2 = np.bincount(shell.ravel(), (np.abs(B) ** 2).ravel(), nbins)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / np.sqrt(d1 * d2)
    return np.nan_to_num(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="RELION's map")
    ap.add_argument("test", help="the map to check")
    ap.add_argument("--apix", type=float, default=None,
                    help="sampling in A/px (default: read from the reference header)")
    args = ap.parse_args()

    ref, apix_ref = read_mrc(args.reference)
    test, _ = read_mrc(args.test)
    apix = args.apix or apix_ref

    print(f"reference {args.reference}  shape={ref.shape}")
    print(f"test      {args.test}  shape={test.shape}")
    if ref.shape != test.shape:
        print("\nshapes differ -- reconstruct at the same box before comparing")
        return 1

    # normalise away any global scale/offset
    z = lambda v: (v - v.mean()) / (v.std() or 1.0)
    a, b = z(ref), z(test)

    cc = float((a * b).mean())
    cc_flip = float((a * z(test[::-1, ::-1, ::-1])).mean())

    # Magnitudes: a negated map makes both correlations negative
    inverted = cc < 0 and abs(cc) > 0.5
    mirrored = abs(cc_flip) > abs(cc) + 0.1

    print(f"\nreal-space correlation      {cc: .4f}"
          + ("   <-- CONTRAST INVERTED (same map, opposite sign)" if inverted else ""))
    print(f"correlation, map mirrored   {cc_flip: .4f}"
          + ("   <-- the test map is MIRRORED" if mirrored else ""))

    c = fsc(a, b)
    n = ref.shape[0]
    print(f"\n{'shell':>6} {'1/A':>10} {'A':>9} {'FSC':>8}")
    for i in range(1, len(c)):
        res = (n * (apix or 1.0)) / i
        unit = "px" if apix is None else "A"
        print(f"{i:>6} {i / (n * (apix or 1.0)):>10.4f} {res:>8.1f}{unit} {c[i]:>8.4f}")

    good = c[1:len(c) - 1]
    med, med_abs = float(np.median(good)), float(np.median(np.abs(good)))
    print(f"\nmedian FSC over all shells  {med: .4f}   (|FSC| {med_abs:.4f})")
    signal = good[np.abs(good) > 0.5]
    if signal.size and np.all(signal < 0):
        print(f"CONTRAST INVERTED but otherwise IDENTICAL: |FSC| {med_abs:.4f} over the "
              f"{signal.size} shells carrying signal.")
        print("      The geometry agrees; only the overall sign differs. Look at the CTF "
              "sign convention, not at the poses.")
    elif med > 0.95:
        print("the two reconstructions agree -- the conversion is exact")
    elif mirrored:
        print("MIRRORED: check rlnTomoHand / the sign of the tilt angles")
    elif good[:len(good) // 4].mean() < 0.5:
        print("disagreement from the lowest shells -- a pose or shift convention "
              "mismatch, not a weighting difference")
    else:
        print("agreement at low frequency, divergence at high -- most likely a CTF "
              "weighting or interpolation difference rather than a wrong convention")
    return 0


if __name__ == "__main__":
    sys.exit(main())
