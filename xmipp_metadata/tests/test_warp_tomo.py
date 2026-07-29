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

import numpy as np
import pandas as pd
import pytest
import starfile

from xmipp_metadata.metadata.warp_tomo import (
    has_warp_tilt_series_labels,
    is_warp_tilt_series_star,
    warp_star_to_tilt_particles,
)


APIX = 2.7
N_PARTICLES = 6
N_TILTS = 9
DOSE_PER_TILT = 3.0


@pytest.fixture
def warp_star(tmp_path):
    """A synthetic Warp-1.x / M particle-series file: one row per tilt image."""
    rng = np.random.default_rng(20)

    tilt_angles = np.linspace(-48, 48, N_TILTS)
    # acquisition order of a dose-symmetric scheme: least tilted first
    acquisition = np.argsort(np.abs(tilt_angles), kind="stable")
    dose = np.zeros(N_TILTS)
    dose[acquisition] = np.arange(N_TILTS) * DOSE_PER_TILT

    rows = []
    for p in range(N_PARTICLES):
        for f in range(N_TILTS):
            rows.append({
                "rlnGroupName": f"TS_1/particle_{p:03d}",
                "rlnImageName": f"{f + 1}@particleseries/TS_1_{p:06d}.mrcs",
                "rlnMicrographName": f"tiltseries/TS_1_{f:03d}.mrc",
                "rlnAngleRot": rng.uniform(-180, 180),
                "rlnAngleTilt": rng.uniform(0, 180),
                "rlnAnglePsi": rng.uniform(-180, 180),
                "rlnOriginXAngst": rng.uniform(-20, 20),
                "rlnOriginYAngst": rng.uniform(-20, 20),
                "rlnDefocusU": rng.uniform(10000, 30000),
                "rlnDefocusV": rng.uniform(10000, 30000),
                "rlnDefocusAngle": rng.uniform(0, 180),
                "rlnCtfScalefactor": np.cos(np.deg2rad(tilt_angles[f])),
                "rlnCtfBfactor": -4.0 * dose[f],
                "rlnOpticsGroup": 1,
            })

    particles = pd.DataFrame(rows)
    # shuffle, so the reader cannot rely on the tilts of a particle being contiguous
    particles = particles.sample(frac=1.0, random_state=7).reset_index(drop=True)

    optics = pd.DataFrame({
        "rlnOpticsGroup": [1],
        "rlnImagePixelSize": [APIX],
        "rlnImageSize": [96],
        "rlnVoltage": [300.0],
        "rlnSphericalAberration": [2.7],
        "rlnAmplitudeContrast": [0.1],
    })

    path = tmp_path / "warp_particles.star"
    starfile.write({"optics": optics, "particles": particles}, path, overwrite=True)
    return dict(path=path, tilt_angles=tilt_angles, dose=dose, particles=particles)


# --------------------------------------------------------------------------- #
#  Detection
# --------------------------------------------------------------------------- #

def test_detects_warp_tilt_series(warp_star):
    blocks = starfile.read(warp_star["path"], always_dict=True)
    assert is_warp_tilt_series_star(blocks)


def test_does_not_mistake_an_spa_star_for_a_tilt_series():
    """An ordinary single-particle file also has rlnGroupNumber -- it must not match."""
    spa = pd.DataFrame({
        "rlnImageName": ["1@a.mrcs", "2@a.mrcs"],
        "rlnGroupNumber": [1, 1],
        "rlnAngleRot": [0.0, 10.0],
        "rlnAngleTilt": [0.0, 10.0],
        "rlnAnglePsi": [0.0, 10.0],
        "rlnOriginXAngst": [0.0, 1.0],
        "rlnOriginYAngst": [0.0, 1.0],
    })
    assert not is_warp_tilt_series_star({"particles": spa})


def _spa_star(tmp_path, n_micrographs=3, n_particles=40, scalefactor=True):
    """
    A RELION single-particle refinement file. CTF::write emits rlnCtfScalefactor, so
    an SPA file carries the labels a Warp tilt series is recognised by.
    """
    rng = np.random.default_rng(3)
    rows = []
    for m in range(n_micrographs):
        for p in range(n_particles):
            row = {
                "rlnImageName": f"{p + 1}@Extract/job005/mic_{m:03d}.mrcs",
                "rlnMicrographName": f"MotionCorr/mic_{m:03d}.mrc",
                "rlnCoordinateX": rng.uniform(0, 4000),
                "rlnCoordinateY": rng.uniform(0, 4000),
                "rlnGroupNumber": m + 1,
                "rlnGroupName": f"group_{m + 1}",
                "rlnAngleRot": rng.uniform(-180, 180),
                "rlnAngleTilt": rng.uniform(0, 180),
                "rlnAnglePsi": rng.uniform(-180, 180),
                "rlnOriginXAngst": rng.uniform(-10, 10),
                "rlnOriginYAngst": rng.uniform(-10, 10),
                "rlnDefocusU": rng.uniform(10000, 30000),
                "rlnDefocusV": rng.uniform(10000, 30000),
                "rlnDefocusAngle": rng.uniform(0, 180),
                "rlnCtfBfactor": rng.uniform(-40, 0),
                "rlnOpticsGroup": 1,
            }
            if scalefactor:
                # relion_ctf_refine --fit_bfac fits this per particle
                row["rlnCtfScalefactor"] = rng.uniform(0.8, 1.0)
            rows.append(row)

    optics = pd.DataFrame({
        "rlnOpticsGroup": [1], "rlnImagePixelSize": [APIX], "rlnImageSize": [96],
        "rlnVoltage": [300.0], "rlnSphericalAberration": [2.7],
        "rlnAmplitudeContrast": [0.1],
    })
    particles = pd.DataFrame(rows)
    path = tmp_path / "spa_run_data.star"
    starfile.write({"optics": optics, "particles": particles}, path, overwrite=True)
    return dict(path=path, particles=particles, n=n_micrographs * n_particles)


def test_does_not_mistake_a_ctf_refined_spa_star_for_a_tilt_series(tmp_path):
    """rlnCtfScalefactor plus rlnGroupNumber is exactly what an SPA file looks like."""
    spa = _spa_star(tmp_path)
    blocks = starfile.read(spa["path"], always_dict=True)
    assert has_warp_tilt_series_labels(blocks["particles"].columns)
    assert not is_warp_tilt_series_star(blocks)


def test_spa_star_is_not_auto_expanded(tmp_path):
    from xmipp_metadata.metadata import XmippMetaData

    spa = _spa_star(tmp_path)
    assert XmippMetaData._sniffTomoKind(str(spa["path"])) is None

    md = XmippMetaData(str(spa["path"]))
    assert not md.isTomo and md.tomoFormat is None
    assert len(md) == spa["n"]
    assert not md.isMetaDataLabel("subtomo_labels")


def test_labels_are_scoped_to_their_block(tmp_path):
    """The grouping column alone in one block must not complete a match in another."""
    from xmipp_metadata.metadata import XmippMetaData

    particles = pd.DataFrame({
        "rlnImageName": ["1@a.mrcs", "2@a.mrcs"],
        "rlnCtfScalefactor": [1.0, 1.0],
        "rlnAngleRot": [0.0, 10.0],
    })
    groups = pd.DataFrame({"rlnGroupNumber": [1, 2], "rlnGroupScaleCorrection": [1.0, 1.0]})
    path = tmp_path / "split.star"
    starfile.write({"particles": particles, "model_groups": groups}, path, overwrite=True)

    assert XmippMetaData._sniffTomoKind(str(path)) is None


def test_does_not_claim_a_relion5_file():
    relion5 = pd.DataFrame({
        "rlnTomoName": ["ts_1"],
        "rlnTomoVisibleFrames": ["[1,1,1]"],
        "rlnCtfScalefactor": [1.0],
        "rlnGroupNumber": [1],
    })
    assert not is_warp_tilt_series_star({"particles": relion5})


# --------------------------------------------------------------------------- #
#  Conversion
# --------------------------------------------------------------------------- #

def test_subtomo_labels_group_the_tilts(warp_star):
    df = warp_star_to_tilt_particles(warp_star["path"])

    assert len(df) == N_PARTICLES * N_TILTS
    labels = df["subtomo_labels"].to_numpy()
    assert labels.dtype.kind in "iu"
    assert np.array_equal(np.unique(labels), np.arange(1, N_PARTICLES + 1))
    assert np.all(np.bincount(labels)[1:] == N_TILTS)

    # a label must correspond to exactly one rlnGroupName
    for label in range(1, N_PARTICLES + 1):
        names = set(df[df["subtomo_labels"] == label]["rlnGroupName"])
        assert len(names) == 1


def test_labels_are_numbered_by_first_appearance(warp_star):
    df = warp_star_to_tilt_particles(warp_star["path"])
    seen, expected = {}, []
    for name in df["rlnGroupName"]:
        if name not in seen:
            seen[name] = len(seen) + 1
        expected.append(seen[name])
    assert np.array_equal(df["subtomo_labels"].to_numpy(), np.asarray(expected))


def test_tilt_ordering_by_scalefactor(warp_star):
    """Ranking rlnCtfScalefactor descending recovers the dose-symmetric order."""
    df = warp_star_to_tilt_particles(warp_star["path"], sort_tilts="scalefactor")

    for label in range(1, N_PARTICLES + 1):
        rows = df[df["subtomo_labels"] == label].sort_values("rlnTomoFrameIndex")
        scale = rows["rlnCtfScalefactor"].to_numpy()
        assert np.all(np.diff(scale) <= 1e-12), "not sorted by scalefactor descending"
        assert np.array_equal(np.sort(rows["rlnTomoFrameIndex"].to_numpy()),
                              np.arange(1, N_TILTS + 1))


def test_tilt_ordering_by_bfactor(warp_star):
    df = warp_star_to_tilt_particles(warp_star["path"], sort_tilts="bfactor")
    for label in range(1, N_PARTICLES + 1):
        rows = df[df["subtomo_labels"] == label].sort_values("rlnTomoFrameIndex")
        # B factor descending == accumulated dose ascending
        assert np.all(np.diff(rows["rlnCtfBfactor"].to_numpy()) <= 1e-12)
        assert np.all(np.diff(rows["rlnMicrographPreExposure"].to_numpy()) >= -1e-12)


def test_file_order_is_preserved_when_asked(warp_star):
    df = warp_star_to_tilt_particles(warp_star["path"], sort_tilts="file")
    original = warp_star["particles"]
    assert list(df["rlnImageName"]) == list(original["rlnImageName"])
    for label in range(1, N_PARTICLES + 1):
        # rows keep their file order, so the ranks just count up within the group
        ranks = df[df["subtomo_labels"] == label]["rlnTomoFrameIndex"].to_numpy()
        assert np.array_equal(ranks, np.arange(1, len(ranks) + 1))


def test_dose_recovered_from_bfactor(warp_star):
    df = warp_star_to_tilt_particles(warp_star["path"])
    assert "rlnMicrographPreExposure" in df.columns
    assert np.allclose(df["rlnMicrographPreExposure"],
                       -df["rlnCtfBfactor"] / 4.0, atol=1e-9)
    # and it must reproduce the dose the fixture encoded
    assert np.isclose(df["rlnMicrographPreExposure"].max(),
                      (N_TILTS - 1) * DOSE_PER_TILT)


def test_existing_dose_column_is_not_overwritten(warp_star, tmp_path):
    blocks = starfile.read(warp_star["path"], always_dict=True)
    blocks["particles"]["rlnMicrographPreExposure"] = 99.0
    path = tmp_path / "with_dose.star"
    starfile.write(blocks, path, overwrite=True)

    df = warp_star_to_tilt_particles(path)
    assert np.allclose(df["rlnMicrographPreExposure"], 99.0)


def test_shift_units_conversion(warp_star):
    angst = warp_star_to_tilt_particles(warp_star["path"])
    px = warp_star_to_tilt_particles(warp_star["path"], shift_units="pixel")

    assert "rlnOriginXAngst" in angst.columns
    assert "rlnOriginX" in px.columns and "rlnOriginXAngst" not in px.columns
    assert np.allclose(px["rlnOriginX"] * APIX, angst["rlnOriginXAngst"], atol=1e-9)
    assert np.allclose(px["rlnOriginY"] * APIX, angst["rlnOriginYAngst"], atol=1e-9)


def test_ragged_tilt_counts_warn(warp_star, tmp_path):
    blocks = starfile.read(warp_star["path"], always_dict=True)
    parts = blocks["particles"]
    victim = parts["rlnGroupName"].iloc[0]
    blocks["particles"] = parts[
        ~((parts["rlnGroupName"] == victim)
          & (parts["rlnCtfScalefactor"] == parts["rlnCtfScalefactor"].min()))
    ].reset_index(drop=True)
    path = tmp_path / "ragged.star"
    starfile.write(blocks, path, overwrite=True)

    with pytest.warns(RuntimeWarning, match="tilt images"):
        df = warp_star_to_tilt_particles(path)
    assert len(df) < N_PARTICLES * N_TILTS

    with pytest.raises(ValueError, match="tilt images"):
        warp_star_to_tilt_particles(path, require_uniform_tilts=True)


def test_missing_group_column_is_an_error(tmp_path):
    parts = pd.DataFrame({
        "rlnImageName": ["1@a.mrcs"],
        "rlnCtfScalefactor": [1.0],
        "rlnAngleRot": [0.0],
    })
    path = tmp_path / "nogroup.star"
    starfile.write({"particles": parts}, path, overwrite=True)

    with pytest.raises(ValueError, match="no column found to group"):
        warp_star_to_tilt_particles(path)


# --------------------------------------------------------------------------- #
#  XmippMetaData integration
# --------------------------------------------------------------------------- #

def test_xmipp_metadata_reads_warp_star(warp_star):
    from xmipp_metadata.metadata import XmippMetaData

    md = XmippMetaData(str(warp_star["path"]))
    assert md.isTomo
    assert md.tomoFormat == "warp"
    assert len(md) == N_PARTICLES * N_TILTS

    labels = md.getMetaDataLabels()
    for expected in ("angleRot", "angleTilt", "anglePsi", "shiftX", "shiftY",
                     "subtomo_labels", "ctfDefocusU", "image"):
        assert expected in labels, expected

    subtomo = md.getMetaDataColumns("subtomo_labels").astype(int)
    assert np.array_equal(np.unique(subtomo), np.arange(1, N_PARTICLES + 1))

    # shifts must arrive in pixels, not Angstrom
    reference = warp_star_to_tilt_particles(warp_star["path"], shift_units="pixel")
    assert np.allclose(md.getMetaDataColumns("shiftX"),
                       reference["rlnOriginX"].to_numpy(), atol=1e-9)


def test_xmipp_metadata_warp_can_be_forced(warp_star):
    from xmipp_metadata.metadata import XmippMetaData

    md = XmippMetaData(str(warp_star["path"]), tomo="warp")
    assert md.tomoFormat == "warp"

    md = XmippMetaData(str(warp_star["path"]), tomo=False)
    assert not md.isTomo
    assert md.tomoFormat is None
    assert len(md) == N_PARTICLES * N_TILTS  # unexpanded, but already one row per tilt
