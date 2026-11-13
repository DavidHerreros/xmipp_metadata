#!/usr/bin/env python
# **************************************************************************
# *
# * Authors:     David Herreros (dherreros@cnb.csic.es)
# *
# * National Centre for Biotechnology (CSIC), Spain
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

"""
Round-trips a RELION-3.1-style STAR file through XmippMetaData and back, checking that
the labels Scipion/RELION actually key CTF and shift detection on survive the trip:
rlnDefocusU/V/Angle (not rlnCtfDefocusU/V/Angle), rlnOriginXAngst/YAngst (not the bare
pixel labels once a pixel size is known), and the passthrough optics columns that have no
Xmipp equivalent (rlnOpticsGroupName, rlnImagePixelSize, rlnImageSize, ...).
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest
import starfile

from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.utils import xmipp_df_to_relion_labels


APIX = 1.302
N_PARTICLES = 5


def _build_relion31_star(path):
    optics = pd.DataFrame([{
        "rlnOpticsGroupName": "opticsGroup1",
        "rlnOpticsGroup": 1,
        "rlnMicrographOriginalPixelSize": 1.0,
        "rlnVoltage": 300.0,
        "rlnSphericalAberration": 2.7,
        "rlnAmplitudeContrast": 0.1,
        "rlnImageSize": 64,
        "rlnImageDimensionality": 2,
        "rlnImagePixelSize": APIX,
    }])

    rng = np.random.default_rng(0)
    n = N_PARTICLES
    particles = pd.DataFrame({
        "rlnImageName": [f"{i + 1:06d}@fake.mrcs" for i in range(n)],
        "rlnMicrographName": [f"mic_{i + 1:03d}.mrc" for i in range(n)],
        "rlnCoordinateX": rng.uniform(100, 900, n),
        "rlnCoordinateY": rng.uniform(100, 900, n),
        "rlnDefocusU": rng.uniform(10000, 30000, n),
        "rlnDefocusV": rng.uniform(10000, 30000, n),
        "rlnDefocusAngle": rng.uniform(0, 180, n),
        "rlnOriginXAngst": rng.uniform(-10, 10, n),
        "rlnOriginYAngst": rng.uniform(-10, 10, n),
        "rlnOriginZAngst": np.zeros(n),
        "rlnAngleRot": rng.uniform(-180, 180, n),
        "rlnAngleTilt": rng.uniform(0, 180, n),
        "rlnAnglePsi": rng.uniform(-180, 180, n),
        "rlnOpticsGroup": np.ones(n, dtype=int),
        "rlnRandomSubset": (np.arange(n) % 2) + 1,
    })

    starfile.write({"optics": optics, "particles": particles}, path, overwrite=True)
    return optics, particles


def _build_legacy_xmipp_star(path):
    """Write the pre-fix layout that Scipion did not recognise as carrying CTF data."""
    optics = pd.DataFrame([{
        "rlnOpticsGroup": 1,
        "rlnVoltage": 300.0,
        "rlnSphericalAberration": 2.7,
        "rlnAmplitudeContrast": 0.1,
    }])
    particles = pd.DataFrame({
        "rlnImageName": ["000001@fake.mrcs"],
        "rlnOpticsGroup": [1],
        "rlnCtfDefocusU": [12345.0],
        "rlnCtfDefocusV": [12678.0],
        "rlnCtfDefocusAngle": [42.0],
        # These optics fields used to leak into the particles loop.
        "rlnOpticsGroupName": ["opticsGroup1"],
        "rlnMicrographOriginalPixelSize": [1.0],
        "rlnImageSize": [200],
        "rlnImageDimensionality": [2],
        "rlnImagePixelSize": [APIX],
    })
    starfile.write({"optics": optics, "particles": particles}, path, overwrite=True)


@pytest.fixture
def relion31_star(tmp_path):
    path = tmp_path / "relion31_particles.star"
    optics, particles = _build_relion31_star(path)
    return dict(path=path, optics=optics, particles=particles)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_read_maps_defocus_and_shifts_to_pixels(relion31_star):
    md = XmippMetaData(str(relion31_star["path"]))

    particles = relion31_star["particles"]
    got_shift_x = md.getMetaDataColumns("shiftX").ravel()
    got_shift_y = md.getMetaDataColumns("shiftY").ravel()
    assert np.allclose(got_shift_x, particles["rlnOriginXAngst"].to_numpy() / APIX)
    assert np.allclose(got_shift_y, particles["rlnOriginYAngst"].to_numpy() / APIX)

    assert "ctfDefocusU" in md.getMetaDataLabels()
    assert np.allclose(md.getMetaDataColumns("ctfDefocusU").ravel(),
                       particles["rlnDefocusU"].to_numpy())

    # the Angstrom-labelled shift column must not survive the rename
    assert "rlnOriginXAngst" not in md.table.columns
    # the optics passthrough column must reach the Xmipp table
    assert "rlnImagePixelSize" in md.table.columns
    assert np.allclose(pd.to_numeric(md.table["rlnImagePixelSize"]).to_numpy(), APIX)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_write_produces_a_relion_readable_star(relion31_star, tmp_path):
    md = XmippMetaData(str(relion31_star["path"]))

    out_path = tmp_path / "out.star"
    md.write(str(out_path))

    with open(out_path) as f:
        first_line = f.readline().rstrip("\n")
    assert first_line == "# version 30001"

    blocks = starfile.read(out_path, always_dict=True)
    assert set(blocks) >= {"optics", "particles"}
    optics_out, particles_out = blocks["optics"], blocks["particles"]

    assert len(optics_out) == 1
    for col in ("rlnOpticsGroupName", "rlnOpticsGroup", "rlnVoltage",
               "rlnSphericalAberration", "rlnAmplitudeContrast", "rlnImagePixelSize",
               "rlnImageSize", "rlnImageDimensionality", "rlnMicrographOriginalPixelSize"):
        assert col in optics_out.columns, col
    assert pd.api.types.is_integer_dtype(optics_out["rlnImageSize"])

    for col in ("rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle",
               "rlnOriginXAngst", "rlnOriginYAngst"):
        assert col in particles_out.columns, col
    for col in ("rlnCtfDefocusU", "rlnOriginX", "rlnImagePixelSize",
               "rlnOpticsGroupName", "rlnVoltage"):
        assert col not in particles_out.columns, col

    particles_in = relion31_star["particles"]
    assert np.allclose(particles_out["rlnDefocusU"].to_numpy(),
                       particles_in["rlnDefocusU"].to_numpy())
    assert np.allclose(particles_out["rlnOriginXAngst"].to_numpy(),
                       particles_in["rlnOriginXAngst"].to_numpy(), atol=1e-4)
    assert np.allclose(particles_out["rlnOriginYAngst"].to_numpy(),
                       particles_in["rlnOriginYAngst"].to_numpy(), atol=1e-4)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_round_trip_is_stable(relion31_star, tmp_path):
    md = XmippMetaData(str(relion31_star["path"]))
    out_path = tmp_path / "out.star"
    md.write(str(out_path))

    md2 = XmippMetaData(str(out_path))

    for col in ("shiftX", "shiftY", "ctfDefocusU", "ctfDefocusV", "ctfDefocusAngle",
               "angleRot", "angleTilt", "anglePsi"):
        assert np.allclose(md.getMetaDataColumns(col).ravel(),
                           md2.getMetaDataColumns(col).ravel(), atol=1e-4), col


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_in_place_rewrite_works(relion31_star, tmp_path):
    p = tmp_path / "inplace.star"
    shutil.copy(relion31_star["path"], p)

    md = XmippMetaData(str(p))
    md.write(str(p))

    with open(p) as f:
        first_line = f.readline().rstrip("\n")
    assert first_line == "# version 30001"

    blocks = starfile.read(p, always_dict=True)
    assert "rlnDefocusU" in blocks["particles"].columns
    assert "rlnCtfDefocusU" not in blocks["particles"].columns
    assert "rlnOpticsGroupName" in blocks["optics"].columns
    assert len(blocks["optics"]) == 1


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_in_place_rewrite_repairs_legacy_xmipp_layout(tmp_path):
    path = tmp_path / "legacy_xmipp.star"
    _build_legacy_xmipp_star(path)

    XmippMetaData(str(path)).write(str(path))

    blocks = starfile.read(path, always_dict=True)
    optics, particles = blocks["optics"], blocks["particles"]

    assert particles.loc[0, "rlnDefocusU"] == 12345.0
    assert particles.loc[0, "rlnDefocusV"] == 12678.0
    assert particles.loc[0, "rlnDefocusAngle"] == 42.0
    assert "rlnCtfDefocusU" not in particles.columns

    for column in ("rlnOpticsGroupName", "rlnMicrographOriginalPixelSize",
                   "rlnImageSize", "rlnImageDimensionality", "rlnImagePixelSize"):
        assert column in optics.columns
        assert column not in particles.columns


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_legacy_xmd_without_pixel_size_writes_pixel_shifts(tmp_path):
    package_path = os.path.abspath(os.path.dirname(__file__))
    data_test_path = os.path.join(package_path, "data")
    cwd = os.getcwd()
    os.chdir(data_test_path)
    try:
        md = XmippMetaData("input_particles.xmd")
    finally:
        os.chdir(cwd)

    assert "ctfDefocusU" in md.table.columns
    assert "shiftX" in md.table.columns and "shiftY" in md.table.columns
    assert "rlnImagePixelSize" not in md.table.columns

    out_path = tmp_path / "legacy.star"
    md.write(str(out_path))

    blocks = starfile.read(out_path, always_dict=True)
    particles_out, optics_out = blocks["particles"], blocks["optics"]

    assert "rlnDefocusU" in particles_out.columns
    assert "rlnOriginX" in particles_out.columns
    assert "rlnOriginXAngst" not in particles_out.columns

    assert "rlnOpticsGroupName" in optics_out.columns
    assert "rlnVoltage" in optics_out.columns

    with pytest.raises(ValueError):
        xmipp_df_to_relion_labels(md.table, shift_units="angstroms")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
