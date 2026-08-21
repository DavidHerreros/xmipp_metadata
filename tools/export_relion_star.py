#!/usr/bin/env python
"""
Export a tomo STAR file as a per-tilt-image particles STAR that RELION can read.

    python export_relion_star.py run_optimisation_set.star particles_tilts.star

``tomo_star_to_tilt_particles`` expands the tilt series into per-tilt-image rows already
in RELION labels, but it has no notion of a ``data_optics`` block and no opinion on which
of those columns RELION actually needs, so this script picks the columns, builds the
optics block (pixel size, box size, premultiplied flag), and writes the two-block STAR
RELION expects -- it does not go through ``XmippMetaData.write()`` / Xmipp labels at all.

The point of having it is to let RELION reconstruct from the alignment this package
derives. If ``relion_reconstruct`` produces the phantom from these poses, the conversion
is right in RELION's own convention, judged by RELION rather than by us.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import starfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xmipp_metadata.metadata.relion_tomo import tomo_star_to_tilt_particles  # noqa: E402


# Everything RELION needs per particle; anything else is dropped
_PARTICLE_LABELS = [
    "rlnImageName", "rlnMicrographName",
    "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi",
    "rlnOriginXAngst", "rlnOriginYAngst",
    "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle",
    "rlnCtfScalefactor", "rlnPhaseShift", "rlnCtfBfactor",
    "rlnMicrographPreExposure",
    "rlnOpticsGroup", "rlnRandomSubset", "rlnGroupNumber",
]


def _scalar(df, label, default=None):
    if label in df.columns:
        v = np.asarray(df[label].to_numpy()).ravel()
        if v.size:
            return v[0]
    return default


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="RELION-5 particles / optimisation set, or a Warp export")
    ap.add_argument("output", help="per-tilt-image particles STAR for RELION")
    ap.add_argument("--tomograms", default=None)
    ap.add_argument("--shifts", default="auto",
                    choices=("auto", "zero", "residual", "from_origin"))
    ap.add_argument("--box", type=int, default=None,
                    help="rlnImageSize; read from the input's optics block when omitted")
    ap.add_argument("--premultiplied", type=int, default=None, choices=(0, 1),
                    help="override rlnCtfDataAreCtfPremultiplied")
    args = ap.parse_args()

    # shift_units="angstrom" is what RELION reads
    df = tomo_star_to_tilt_particles(args.input, args.tomograms,
                                     shifts=args.shifts, shift_units="angstrom")

    src = starfile.read(args.input, always_dict=True)
    optics_in = src.get("optics")
    if optics_in is None:                       # an optimisation set points elsewhere
        from xmipp_metadata.metadata.relion_tomo import read_optimisation_set
        try:
            src = starfile.read(read_optimisation_set(args.input)["particles"],
                                always_dict=True)
            optics_in = src.get("optics")
        except Exception:
            optics_in = None

    apix = _scalar(df, "rlnImagePixelSize")
    box = args.box or (_scalar(optics_in, "rlnImageSize") if optics_in is not None else None)
    premult = args.premultiplied
    if premult is None:
        premult = _scalar(df, "rlnCtfDataAreCtfPremultiplied",
                          _scalar(optics_in, "rlnCtfDataAreCtfPremultiplied", 0)
                          if optics_in is not None else 0)
    if box is None:
        raise SystemExit("could not determine the box size; pass --box")

    optics = pd.DataFrame([{
        "rlnOpticsGroup": 1,
        "rlnOpticsGroupName": "opticsGroup1",
        "rlnVoltage": float(_scalar(df, "rlnVoltage", 300.0)),
        "rlnSphericalAberration": float(_scalar(df, "rlnSphericalAberration", 2.7)),
        "rlnAmplitudeContrast": float(_scalar(df, "rlnAmplitudeContrast", 0.07)),
        "rlnImagePixelSize": float(apix),
        "rlnImageSize": int(box),
        "rlnImageDimensionality": 2,
        "rlnCtfDataAreCtfPremultiplied": int(premult),
    }])

    parts = pd.DataFrame({c: df[c] for c in _PARTICLE_LABELS if c in df.columns})
    # one optics group, so every row points at it (Warp writes a string here)
    parts["rlnOpticsGroup"] = 1
    if "rlnRandomSubset" not in parts.columns:
        parts["rlnRandomSubset"] = np.arange(1, len(parts) + 1) % 2 + 1

    missing = [c for c in ("rlnImageName", "rlnAngleRot", "rlnOriginXAngst",
                           "rlnDefocusU") if c not in parts.columns]
    if missing:
        raise SystemExit(f"the expansion did not produce {missing}")

    starfile.write({"optics": optics, "particles": parts}, args.output, overwrite=True)
    print(f"wrote {args.output}: {len(parts)} tilt images, "
          f"{int(df['subtomo_labels'].max())} particles")
    print(f"  pixel size {float(apix):.4f} A   box {int(box)}   "
          f"premultiplied {int(premult)}")
    print(f"\n    relion_reconstruct --i {args.output} --o relion.mrc"
          f"{' --ctf' if True else ''}")
    if premult:
        print("    (images are pre-multiplied; that is declared in the optics block)")


if __name__ == "__main__":
    main()
