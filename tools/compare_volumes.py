#!/usr/bin/env python
"""
Compare two reconstructions voxel-by-voxel and by FSC.

    python compare_volumes.py relion.mrc hax.mrc [--apix 4.0]

The point of this script is that it does not care whether the maps are any *good*.
Two reconstructions built from the same particles and the same poses must agree
whatever those poses are, so a collapsed refinement is still a valid test case --
in fact a better one, because a wrong convention cannot hide behind a
recognisable structure.

FSC ~ 1.0 out to Nyquist means the two reconstructors agree. A dip that starts
low and stays low means a convention mismatch; a dip that only appears at high
frequency means a weighting or interpolation difference, which is benign.
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

    # normalise away any global scale/offset: a reconstructor is allowed to differ
    # by a constant, and the FSC is blind to it anyway
    z = lambda v: (v - v.mean()) / (v.std() or 1.0)
    a, b = z(ref), z(test)

    cc = float((a * b).mean())
    print(f"\nreal-space correlation      {cc: .4f}")

    # a global handedness flip is the single most common convention error, and it
    # is invisible to anything that only looks at the power spectrum
    cc_flip = float((a * z(test[::-1, ::-1, ::-1])).mean())
    print(f"correlation, map inverted   {cc_flip: .4f}"
          + ("   <-- the test map is MIRRORED" if cc_flip > cc + 0.1 else ""))

    c = fsc(a, b)
    n = ref.shape[0]
    print(f"\n{'shell':>6} {'1/A':>10} {'A':>9} {'FSC':>8}")
    for i in range(1, len(c)):
        res = (n * (apix or 1.0)) / i
        unit = "px" if apix is None else "A"
        print(f"{i:>6} {i / (n * (apix or 1.0)):>10.4f} {res:>8.1f}{unit} {c[i]:>8.4f}")

    good = c[1:len(c) - 1]
    print(f"\nmedian FSC over all shells  {np.median(good): .4f}")
    if np.median(good) > 0.95:
        print("the two reconstructions agree -- the conversion is exact")
    elif cc_flip > cc + 0.1:
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
