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
Warp 2.x / WarpTools ``ts_export_particles --2d`` output.

Unlike Warp 1.x / M, this writes a RELION-5 *shaped* pair of files, so it goes
through the RELION converter rather than the Warp adapter.  The fixture below
reproduces the exact column set WarpTools emits, taken from
``TiltSeries.ReconstructParticleSeries.cs`` and
``ExportParticlesTiltseries.cs``:

  particles  -- rlnTomoName (with a ``.tomostar`` suffix), rlnTomoParticleId,
                rlnCoordinateX/Y/Z (in *binned* pixels), rlnAngleRot/Tilt/Psi,
                rlnTomoParticleName, rlnOpticsGroup (a *string*), rlnImageName
                (no ``index@`` prefix), rlnOriginX/Y/ZAngst (hard zero),
                rlnTomoVisibleFrames. No optics block.
  tomograms  -- a global block with no rlnTomoTiltSeriesStarFile, followed by
                per-tomogram blocks carrying rlnTomoProjX/Y/Z/W rather than
                rlnTomoXTilt/YTilt/ZRot.
"""

import numpy as np
import pandas as pd
import pytest
import starfile

from xmipp_metadata.metadata.relion_tomo import (
    TiltSeriesGeometry,
    projection_matrix_from_angles,
    read_tomograms_star,
    relion_angles_to_matrix,
    tomo_star_to_tilt_particles,
)


APIX = 4.0          # WarpTools writes the *export* pixel size here
BOX = 64
N_TILTS = 9
N_PARTICLES = 5
W = H = 512
D = 256


def _vec(values):
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


def _write_project(tmp_path, tomo_names=("TS_01.tomostar", "TS_02.tomostar"),
                   block_names=None, stem="matching_4apx", rotation_only=True,
                   refined_origins=False):
    rng = np.random.default_rng(42)

    global_rows, blocks, projections = [], {}, {}
    for ti, tomo in enumerate(tomo_names):
        proj = projection_matrix_from_angles(
            rng.uniform(-2, 2, N_TILTS),
            np.linspace(-45, 45, N_TILTS),
            rng.uniform(-180, 180, N_TILTS),
            rng.uniform(-30, 30, N_TILTS),
            rng.uniform(-30, 30, N_TILTS),
            APIX, (W, H, D), (W, H))
        if rotation_only:
            # WarpTools writes `[{M11},{M12},{M13},0]` and `[0,0,0,1]` -- a pure
            # rotation, with no specimen centre, image centre or per-tilt shift
            proj = proj.copy()
            proj[:, :3, 3] = 0.0
        projections[tomo] = proj

        ts = pd.DataFrame({
            "rlnTomoProjX": [_vec(proj[f, 0]) for f in range(N_TILTS)],
            "rlnTomoProjY": [_vec(proj[f, 1]) for f in range(N_TILTS)],
            "rlnTomoProjZ": [_vec(proj[f, 2]) for f in range(N_TILTS)],
            "rlnTomoProjW": [_vec(proj[f, 3]) for f in range(N_TILTS)],
            "rlnDefocusU": rng.uniform(20000, 40000, N_TILTS),
            "rlnDefocusV": rng.uniform(20000, 40000, N_TILTS),
            "rlnDefocusAngle": rng.uniform(0, 180, N_TILTS),
            "rlnPhaseShift": np.zeros(N_TILTS),
            "rlnCtfScalefactor": np.cos(np.deg2rad(np.linspace(-45, 45, N_TILTS))),
            "rlnMicrographPreExposure": np.arange(N_TILTS) * 3.0,
        })
        key = tomo if block_names is None else block_names[ti]
        blocks[key] = ts

        global_rows.append({
            "rlnTomoName": tomo,
            "rlnTomoTiltSeriesName": f"tiltstack/{tomo[:-9]}/{tomo[:-9]}.mrc",
            "rlnTomoFrameCount": N_TILTS,
            "rlnTomoSizeX": W, "rlnTomoSizeY": H, "rlnTomoSizeZ": D,
            "rlnTomoHand": -1.0,
            "rlnOpticsGroupName": tomo[:-9],
            "rlnTomoTiltSeriesPixelSize": APIX,
            "rlnVoltage": 300.0,
            "rlnSphericalAberration": 2.7,
            "rlnAmplitudeContrast": 0.07,
            "rlnTomoImportFractionalDose": 3.0,
        })

    tomograms_path = tmp_path / f"{stem}_tomograms.star"
    starfile.write({"global": pd.DataFrame(global_rows), **blocks},
                   tomograms_path, overwrite=True)

    rows = []
    for tomo in tomo_names:
        root = tomo[:-9]
        for p in range(N_PARTICLES):
            rows.append({
                "rlnTomoName": tomo,
                "rlnTomoParticleId": p + 1,
                "rlnCoordinateX": rng.uniform(0.35 * W, 0.65 * W),
                "rlnCoordinateY": rng.uniform(0.35 * H, 0.65 * H),
                "rlnCoordinateZ": rng.uniform(0.35 * D, 0.65 * D),
                "rlnAngleRot": rng.uniform(-180, 180),
                "rlnAngleTilt": rng.uniform(0, 180),
                "rlnAnglePsi": rng.uniform(-180, 180),
                "rlnTomoParticleName": f"{root}/{p + 1}",
                "rlnOpticsGroup": root,                     # a string, as Warp writes
                "rlnImageName": f"particleseries/{root}_{p + 1:06d}.mrcs",
                # WarpTools writes these as a hard 0.0; a refinement run on top of the
                # exported stacks is what makes them non-zero
                "rlnOriginXAngst": rng.uniform(-12, 12) if refined_origins else 0.0,
                "rlnOriginYAngst": rng.uniform(-12, 12) if refined_origins else 0.0,
                "rlnOriginZAngst": rng.uniform(-12, 12) if refined_origins else 0.0,
                "rlnTomoVisibleFrames": _vec(np.ones(N_TILTS)).replace(".000000", ""),
            })

    particles_path = tmp_path / f"{stem}.star"
    starfile.write({"particles": pd.DataFrame(rows)}, particles_path, overwrite=True)

    return dict(particles=particles_path, tomograms=tomograms_path,
                projections=projections, tomo_names=list(tomo_names))


@pytest.fixture
def warp2_project(tmp_path):
    return _write_project(tmp_path)


def test_tomograms_star_without_tiltseriesstarfile(warp2_project):
    """WarpTools uses the inline layout and rlnTomoProjX/Y/Z/W, not the Euler form."""
    geoms = read_tomograms_star(warp2_project["tomograms"])
    assert set(geoms) == set(warp2_project["tomo_names"])

    geom = geoms["TS_01.tomostar"]
    assert geom.frame_count == N_TILTS
    assert geom.pixel_size == APIX
    assert geom.handedness == -1.0
    # the matrices must come back exactly as written, not rebuilt from angles
    assert np.allclose(geom.projection, warp2_project["projections"]["TS_01.tomostar"],
                       atol=1e-6)


def test_tomograms_blocks_matched_by_position(tmp_path):
    """
    RELION matches the inline blocks positionally (allTables[t+1]), so a block whose
    name does not equal rlnTomoName must still be found.
    """
    project = _write_project(tmp_path, block_names=["tilt_series_1", "tilt_series_2"])
    geoms = read_tomograms_star(project["tomograms"])
    assert set(geoms) == set(project["tomo_names"])
    assert np.allclose(geoms["TS_02.tomostar"].projection,
                       project["projections"]["TS_02.tomostar"], atol=1e-6)


def test_tomograms_star_is_auto_discovered(warp2_project):
    """WarpTools names it <prefix>_tomograms.star, not tomograms.star."""
    df = tomo_star_to_tilt_particles(warp2_project["particles"])
    assert len(df) == 2 * N_PARTICLES * N_TILTS


def test_conversion_matches_the_warp_geometry(warp2_project):
    df = tomo_star_to_tilt_particles(warp2_project["particles"],
                                     warp2_project["tomograms"])

    n = 2 * N_PARTICLES
    assert len(df) == n * N_TILTS
    assert np.array_equal(np.unique(df["subtomo_labels"]), np.arange(1, n + 1))

    # rlnTomoVisibleFrames is present, so "auto" must centre the images
    assert np.allclose(df["rlnOriginXAngst"], 0.0)
    assert np.allclose(df["rlnOriginYAngst"], 0.0)

    parts = starfile.read(warp2_project["particles"], always_dict=True)["particles"]
    # Compare against the matrices as *read back*. Warp writes rlnTomoProj* with six
    # decimals, so they are not exactly orthonormal, and an Euler triplet can only
    # represent a proper rotation -- the pose is therefore the nearest rotation to
    # R_f @ A_part. That residual is a property of the file format, not of the
    # conversion, so the test bounds it by the matrices' own non-orthonormality.
    proj = read_tomograms_star(warp2_project["tomograms"])["TS_01.tomostar"].projection
    R = proj[:, :3, :3]
    non_orthonormality = np.abs(R @ np.swapaxes(R, -1, -2) - np.eye(3)).max()
    assert non_orthonormality < 1e-5, "fixture matrices are further off than expected"

    for label in (1, 3, 5):
        p = parts.iloc[label - 1]
        A_part = relion_angles_to_matrix(p["rlnAngleRot"], p["rlnAngleTilt"],
                                         p["rlnAnglePsi"])
        rows = df[df["subtomo_labels"] == label].sort_values("rlnTomoFrameIndex")
        for i, (_, r) in enumerate(rows.iterrows()):
            got = relion_angles_to_matrix(r["rlnAngleRot"], r["rlnAngleTilt"],
                                          r["rlnAnglePsi"])
            err = np.abs(got - R[i] @ A_part).max()
            assert err <= 4 * non_orthonormality, (label, i, err)


def test_stack_indices_for_warp_image_names(warp2_project):
    """Warp writes the series path with no index@ prefix; one slice per visible tilt."""
    df = tomo_star_to_tilt_particles(warp2_project["particles"],
                                     warp2_project["tomograms"])
    rows = df[df["subtomo_labels"] == 1].sort_values("rlnTomoFrameIndex")
    names = list(rows["rlnImageName"])
    assert [s.split("@")[0] for s in names] == [str(i + 1) for i in range(N_TILTS)]
    assert all(s.split("@")[1] == "particleseries/TS_01_000001.mrcs" for s in names)


def test_string_optics_group_survives(warp2_project):
    """rlnOpticsGroup is a tomogram name string in Warp output, not an integer."""
    df = tomo_star_to_tilt_particles(warp2_project["particles"],
                                     warp2_project["tomograms"])
    assert set(df["rlnOpticsGroup"]) == {"TS_01", "TS_02"}


def test_binning_defaults_to_one_without_an_optics_block(warp2_project):
    """
    Warp writes no optics block, so there is no rlnImagePixelSize to derive a binning
    from. The coordinates are already on the export grid that rlnTomoTiltSeriesPixelSize
    describes, so binning must stay 1 and the output sampling must equal it.
    """
    df = tomo_star_to_tilt_particles(warp2_project["particles"],
                                     warp2_project["tomograms"])
    assert np.allclose(df["rlnImagePixelSize"], APIX)
    assert np.allclose(df["rlnTomoTiltSeriesPixelSize"], APIX)


def test_xmipp_metadata_autodetects_warp2(warp2_project):
    from xmipp_metadata.metadata import XmippMetaData

    md = XmippMetaData(str(warp2_project["particles"]))
    assert md.isTomo
    # Warp 2 output is RELION-5 shaped, so it must take the RELION path
    assert md.tomoFormat == "relion"
    assert len(md) == 2 * N_PARTICLES * N_TILTS
    assert "subtomo_labels" in md.getMetaDataLabels()


def test_warp_matrices_are_rotation_only(warp2_project):
    """
    WarpTools writes rlnTomoProj* as a pure rotation. That still gives the right
    orientation, but nothing that depends on an absolute projected position.
    """
    geoms = read_tomograms_star(warp2_project["tomograms"])
    assert all(g.is_rotation_only for g in geoms.values())
    assert np.allclose(geoms["TS_01.tomostar"].projection[:, :3, 3], 0.0)


def test_defocus_gradient_survives_a_missing_translation(warp2_project):
    """
    getDepthOffset is a *difference* of projected depths, so a translation common to
    both cancels: the per-particle defocus is still right on Warp output.
    """
    geoms = read_tomograms_star(warp2_project["tomograms"])
    geom = geoms["TS_01.tomostar"]

    full = np.array(geom.projection, copy=True)
    full[:, :3, 3] = [123.0, -45.0, 7.0]        # any translation at all
    shifted = TiltSeriesGeometry(**{**geom.__dict__, "projection": full})

    pos = np.array([[0.4 * W, 0.55 * H, 0.6 * D], [0.5 * W, 0.5 * H, 0.5 * D]])
    assert np.allclose(geom.defocus_offset(geom.project(pos)),
                       shifted.defocus_offset(shifted.project(pos)), atol=1e-9)


def test_residual_shifts_are_refused_on_rotation_only_geometry(warp2_project):
    with pytest.raises(ValueError, match="zero translation column"):
        tomo_star_to_tilt_particles(warp2_project["particles"],
                                    warp2_project["tomograms"],
                                    shifts="residual")


def test_computed_visibility_is_refused_on_rotation_only_geometry(warp2_project):
    with pytest.raises(ValueError, match="visibility cannot be recomputed"):
        tomo_star_to_tilt_particles(warp2_project["particles"],
                                    warp2_project["tomograms"],
                                    visibility="computed", box_size=BOX)


def test_meaningless_coordinates_are_flagged(warp2_project):
    with pytest.warns(RuntimeWarning, match="not real tilt-image coordinates"):
        tomo_star_to_tilt_particles(warp2_project["particles"],
                                    warp2_project["tomograms"])


def test_full_matrices_do_not_trigger_the_guard(tmp_path):
    """A RELION-written tomograms.star carries the full affine and must be unaffected."""
    project = _write_project(tmp_path, rotation_only=False)
    geoms = read_tomograms_star(project["tomograms"])
    assert not any(g.is_rotation_only for g in geoms.values())
    df = tomo_star_to_tilt_particles(project["particles"], project["tomograms"],
                                     shifts="residual")
    assert len(df) == 2 * N_PARTICLES * N_TILTS


def test_refined_origins_on_warp_geometry(tmp_path):
    """
    The real downstream case: RELION refines on top of Warp's exported 2D stacks, so the
    origins become non-zero while the geometry stays rotation-only. shifts='from_origin'
    must work there, and must be what the default picks -- this is the configuration where
    the old default silently discarded the whole translational refinement.
    """
    project = _write_project(tmp_path, refined_origins=True)

    # the default has to keep the refinement, without being asked
    default = tomo_star_to_tilt_particles(project["particles"], project["tomograms"])
    assert np.abs(default["rlnOriginXAngst"]).max() > 1e-3

    # asking for 'zero' still drops it, and still has to say so
    with pytest.warns(RuntimeWarning, match="shifts='from_origin'"):
        dropped = tomo_star_to_tilt_particles(project["particles"], project["tomograms"],
                                              shifts="zero")
    assert np.allclose(dropped["rlnOriginXAngst"], 0.0)

    # 'residual' is still refused: it needs absolute positions this geometry cannot give
    with pytest.raises(ValueError, match="zero translation column"):
        tomo_star_to_tilt_particles(project["particles"], project["tomograms"],
                                    shifts="residual")

    # 'from_origin' works, because it only ever projects a difference
    kept = tomo_star_to_tilt_particles(project["particles"], project["tomograms"],
                                       shifts="from_origin")
    pd.testing.assert_frame_equal(default, kept)
    assert np.abs(kept["rlnOriginXAngst"]).max() > 1e-3
    assert np.abs(kept["rlnOriginYAngst"]).max() > 1e-3

    # and it agrees with a direct projection of the refined 3D offset
    geoms = read_tomograms_star(project["tomograms"])
    parts = starfile.read(project["particles"], always_dict=True)["particles"]
    for name, geom in geoms.items():
        sub = parts[parts["rlnTomoName"] == name].reset_index(drop=True)
        offset = np.stack([sub["rlnOriginXAngst"].to_numpy(),
                           sub["rlnOriginYAngst"].to_numpy(),
                           sub["rlnOriginZAngst"].to_numpy()], axis=-1)
        delta = -offset / APIX
        want = -np.einsum('fij,nj->nfi', geom.projection[:, :3, :3], delta)[..., :2] * APIX
        rows = kept[kept["rlnTomoName"] == name]
        assert np.allclose(rows["rlnOriginXAngst"].to_numpy(), want[..., 0].ravel(),
                           atol=1e-9)
        assert np.allclose(rows["rlnOriginYAngst"].to_numpy(), want[..., 1].ravel(),
                           atol=1e-9)

    # the poses are untouched by the shift mode
    for col in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"):
        assert np.allclose(dropped[col].to_numpy(), kept[col].to_numpy(), atol=1e-12)


def test_from_origin_reaches_xmipp_metadata(tmp_path):
    """The mode has to be reachable through the XmippMetaData entry point too."""
    from xmipp_metadata.metadata import XmippMetaData

    project = _write_project(tmp_path, refined_origins=True)
    md = XmippMetaData(str(project["particles"]),
                       tomo_kwargs={"shifts": "from_origin"})
    assert md.isTomo and md.tomoFormat == "relion"
    # shift_units defaults to "pixel" on this path, so shiftX is in output pixels
    assert np.abs(md.getMetaDataColumns("shiftX")).max() > 1e-4


def test_missing_tomograms_star_gives_a_clear_error(tmp_path):
    project = _write_project(tmp_path)
    project["tomograms"].unlink()
    with pytest.raises(ValueError, match="no tomograms STAR file given"):
        tomo_star_to_tilt_particles(project["particles"])
