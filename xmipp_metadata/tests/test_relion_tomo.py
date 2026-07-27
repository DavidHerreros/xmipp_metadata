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
The tests below are deliberately written as a *second, independent* transcription
of the RELION routines involved (``Euler_angles2matrix``, ``Tomogram::setProjectionMatrix``,
``ParticleSet::getMatrix4x4``, ``TomoExtraction::extractAt2D_Fourier`` and the
``projPart = scaleRatio * projCut * particleToTomo`` line of
``reconstruct_particle.cpp``), so that they check the conversion against RELION's
algebra rather than against itself.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import starfile

from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.metadata.relion_tomo import (
    relion_angles_to_matrix,
    matrix_to_relion_angles,
    projection_matrix_from_angles,
    read_tomograms_star,
    read_trajectories_star,
    tomo_star_to_tilt_particles,
    read_optimisation_set,
    is_relion_tomo_star,
    _read_star,
    Linear2DDeformation,
    Spline2DDeformation,
    Fourier2DDeformation,
)


# --------------------------------------------------------------------------- #
#  Literal transcriptions of RELION, used as the reference
# --------------------------------------------------------------------------- #

def _eye4():
    return np.eye(4, dtype=np.float64)


def _translation4(t):
    M = _eye4()
    M[:3, 3] = t
    return M


def _gravis_rotation(axis, angle_deg):
    """gravis t3Matrix::rotation -- Rodrigues, angle in DEGREES."""
    n = np.asarray(axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    a = np.deg2rad(angle_deg)
    S = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    nnt = np.outer(n, n)
    R = nnt + np.cos(a) * (np.eye(3) - nnt) + np.sin(a) * S
    M = _eye4()
    M[:3, :3] = R
    return M


def _relion_set_projection_matrix(xtilt, ytilt, zrot, xshift_angst, yshift_angst,
                                  pixel_size, w0, h0, d0, iw, ih):
    """Tomogram::setProjectionMatrix, transcribed line by line."""
    s0 = _translation4(-np.array([int(w0) // 2, int(h0) // 2, int(d0) // 2], float))
    s1 = _translation4([xshift_angst / pixel_size, yshift_angst / pixel_size, 0.0])
    s2 = _translation4([int(iw) // 2, int(ih) // 2, 0.0])
    r0 = _gravis_rotation((1, 0, 0), xtilt)
    r1 = _gravis_rotation((0, 1, 0), ytilt)
    r2 = _gravis_rotation((0, 0, 1), zrot)
    return s1 @ s2 @ r2 @ r1 @ r0 @ s0


def _relion_euler_angles2matrix(alpha, beta, gamma):
    """Euler_angles2matrix, transcribed line by line (degrees in)."""
    alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])
    ca, cb, cg = np.cos([alpha, beta, gamma])
    sa, sb, sg = np.sin([alpha, beta, gamma])
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    return np.array([
        [cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
        [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
        [sc, ss, cb],
    ])


def _relion_get_matrix_4x4(A, pos, s):
    """ParticleSet::getMatrix4x4 -- Ts * R * Tc."""
    Tc = _translation4(-np.array([int(s) // 2] * 3, dtype=np.float64))
    R = _eye4()
    R[:3, :3] = A
    Ts = _translation4(pos)
    return Ts @ R @ Tc


def _relion_proj_cut(P, pos, s, bin_):
    """The projOut update of TomoExtraction::extractAt2D_Fourier."""
    sb = int(s / bin_ + 0.5)
    centre = (P @ np.append(pos, 1.0))[:2]
    out = P.copy()
    out[0, 3] += sb / 2 - centre[0]
    out[1, 3] += sb / 2 - centre[1]
    return out, sb, centre


def _ref_linear_deformation(pl, image_size, c):
    """Linear2DDeformationModel::computeShift, transcribed."""
    centre = np.array([0.5 * image_size[0], 0.5 * image_size[1]])
    r = np.asarray(pl, float) - centre
    return np.asarray(pl, float) + np.array([c[0] * r[0], c[1] * r[0] + c[2] * r[1]])


def _ref_spline_deformation(pl, image_size, grid, coeffs):
    """Spline2DDeformationModel::computeShift + projectPoint, transcribed."""
    gx, gy = int(grid[0]), int(grid[1])
    spacing = (image_size[0] / (gx - 1.0), image_size[1] / (gy - 1.0))

    g = [pl[0] / spacing[0], pl[1] / spacing[1]]
    eps = 1e-10
    for d, n in enumerate((gx, gy)):
        g[d] = 0.0 if g[d] < 0.0 else (n - 1 - eps if g[d] > n - 1 - eps else g[d])
    cell = (int(g[0]), int(g[1]))
    x, y = g[0] - cell[0], g[1] - cell[1]
    x2, x3, y2, y3 = x * x, x * x * x, y * y, y * y * y

    vx = np.array([1.0 - 3 * x2 + 2 * x3, 3 * x2 - 2 * x3, x - 2 * x2 + x3, -x2 + x3])
    vy = np.array([1.0 - 3 * y2 + 2 * y3, 3 * y2 - 2 * y3, y - 2 * y2 + y3, -y2 + y3])

    def node(xx, yy, dim):  # RawImage<DataPoint>(gx, gy, 2), DataPoint = 4 doubles
        base = 4 * (xx + yy * gx + dim * gx * gy)
        return coeffs[base:base + 4]  # value, slope_x, slope_y, twist

    out = np.zeros(2)
    for dim in range(2):
        d00, d01 = node(cell[0], cell[1], dim), node(cell[0], cell[1] + 1, dim)
        d10, d11 = node(cell[0] + 1, cell[1], dim), node(cell[0] + 1, cell[1] + 1, dim)
        F = np.array([
            [d00[0], d01[0], d00[2], d01[2]],
            [d10[0], d11[0], d10[2], d11[2]],
            [d00[1], d01[1], d00[3], d01[3]],
            [d10[1], d11[1], d10[3], d11[3]],
        ])
        out[dim] = vx @ (F @ vy)
    return np.asarray(pl, float) + out


def _ref_fourier_deformation(pl, image_size, grid, coeffs):
    """Fourier2DDeformationModel constructor + computeShift, transcribed."""
    gx, gy = int(grid[0]), int(grid[1])
    freqs = []
    for yy in range(gy):
        if yy < gy // 2:
            for xx in range(1 if yy == 0 else 0, gx // 2 + 1):
                freqs.append((xx * np.pi / image_size[0], yy * np.pi / image_size[1]))
        else:
            for xx in range(1, gx // 2 + 1):
                freqs.append((xx * np.pi / image_size[0],
                              (yy - gy) * np.pi / image_size[1]))

    n = len(freqs)
    out = np.zeros(2)
    for dim in range(2):
        for i, fq in enumerate(freqs):
            t = fq[0] * pl[0] + fq[1] * pl[1]
            re, im = coeffs[2 * (i + dim * n)], coeffs[2 * (i + dim * n) + 1]
            out[dim] += re * np.cos(t) + im * np.sin(t)
    return np.asarray(pl, float) + out


# --------------------------------------------------------------------------- #
#  Euler conventions
# --------------------------------------------------------------------------- #

def test_angles_to_matrix_matches_relion():
    rng = np.random.default_rng(0)
    for _ in range(200):
        rot, psi = rng.uniform(-180, 180, 2)
        tilt = rng.uniform(0, 180)
        assert np.allclose(relion_angles_to_matrix(rot, tilt, psi),
                           _relion_euler_angles2matrix(rot, tilt, psi), atol=1e-12)


def test_angles_matrix_roundtrip():
    rng = np.random.default_rng(1)
    rot = rng.uniform(-180, 180, 500)
    tilt = rng.uniform(0, 180, 500)
    psi = rng.uniform(-180, 180, 500)

    A = relion_angles_to_matrix(rot, tilt, psi)
    r2, t2, p2 = matrix_to_relion_angles(A)
    # the angles may be a different (equivalent) representative, so compare matrices
    assert np.allclose(relion_angles_to_matrix(r2, t2, p2), A, atol=1e-10)
    assert np.all(t2 >= -1e-9) and np.all(t2 <= 180 + 1e-9)


def test_angles_matrix_roundtrip_gimbal_lock():
    rot = np.array([0.0, 37.0, -120.0, 5.0])
    tilt = np.array([0.0, 0.0, 180.0, 180.0])
    psi = np.array([10.0, -73.0, 44.0, 0.0])

    A = relion_angles_to_matrix(rot, tilt, psi)
    r2, t2, p2 = matrix_to_relion_angles(A)
    assert np.allclose(relion_angles_to_matrix(r2, t2, p2), A, atol=1e-10)


def test_matrices_are_rotations():
    rng = np.random.default_rng(2)
    A = relion_angles_to_matrix(rng.uniform(-180, 180, 100),
                                rng.uniform(0, 180, 100),
                                rng.uniform(-180, 180, 100))
    assert np.allclose(A @ np.swapaxes(A, -1, -2), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(A), 1.0, atol=1e-12)


# --------------------------------------------------------------------------- #
#  Tilt-series geometry
# --------------------------------------------------------------------------- #

def test_projection_matrix_matches_relion():
    rng = np.random.default_rng(3)
    apix, w0, h0, d0, iw, ih = 1.35, 4000, 4000, 2000, 4000, 4000

    xtilt = rng.uniform(-3, 3, 41)
    ytilt = np.linspace(-60, 60, 41)
    zrot = rng.uniform(-180, 180, 41)
    xs = rng.uniform(-200, 200, 41)
    ys = rng.uniform(-200, 200, 41)

    P = projection_matrix_from_angles(xtilt, ytilt, zrot, xs, ys,
                                      apix, (w0, h0, d0), (iw, ih))
    for f in range(41):
        ref = _relion_set_projection_matrix(xtilt[f], ytilt[f], zrot[f], xs[f], ys[f],
                                            apix, w0, h0, d0, iw, ih)
        assert np.allclose(P[f], ref, atol=1e-9), f"frame {f}"


def test_composition_is_a_pure_rotation_about_the_box_centre():
    """
    The core claim: projPart_f = scaleRatio * projCut_f * particleToTomo reduces to
    u -> scaleRatio * (A_tot @ (u - s/2) + sb/2) in xy, with A_tot = R_f @ A_sub @ A_part.
    If that holds, the per-tilt pose is exactly matrix_to_relion_angles(A_tot) with a
    zero shift, which is what the converter emits.
    """
    rng = np.random.default_rng(4)
    apix, w0, h0, d0 = 1.35, 4000, 4000, 2000
    s, bin_ = 192, 2.0
    scale_ratio = 1.0

    P = projection_matrix_from_angles(
        rng.uniform(-3, 3, 30), np.linspace(-60, 60, 30), rng.uniform(-180, 180, 30),
        rng.uniform(-200, 200, 30), rng.uniform(-200, 200, 30),
        apix, (w0, h0, d0), (w0, h0))

    for _ in range(20):
        pos = np.array([rng.uniform(500, 3500), rng.uniform(500, 3500),
                        rng.uniform(400, 1600)])
        A_sub = _relion_euler_angles2matrix(*rng.uniform(-180, 180, 3))
        A_part = _relion_euler_angles2matrix(rng.uniform(-180, 180),
                                             rng.uniform(0, 180),
                                             rng.uniform(-180, 180))
        A = A_sub @ A_part
        particle_to_tomo = _relion_get_matrix_4x4(A, pos, s)

        for f in range(30):
            proj_cut, sb, _ = _relion_proj_cut(P[f], pos, s, bin_)
            proj_part = scale_ratio * (proj_cut @ particle_to_tomo)

            A_tot = P[f][:3, :3] @ A

            # a few arbitrary points inside the box
            for u in rng.uniform(0, s, size=(5, 3)):
                got = (proj_part @ np.append(u, 1.0))[:2]
                want = scale_ratio * (A_tot @ (u - s // 2))[:2] + scale_ratio * sb / 2
                assert np.allclose(got, want, atol=1e-8)

            # and the pose we emit reproduces A_tot exactly
            r, t, p = matrix_to_relion_angles(A_tot)
            assert np.allclose(relion_angles_to_matrix(r, t, p), A_tot, atol=1e-10)


def test_origin_shift_is_subtracted_from_the_coordinate():
    """
    ParticleSet::getPosition subtracts the (rotated) origin from the coordinate.
    The converter must use the same sign, or every particle lands two shifts away.
    """
    apix = 1.35
    centre = np.array([2000.0, 2000.0, 1000.0])
    coord_angst = np.array([100.0, -250.0, 40.0])
    origin_angst = np.array([3.0, -7.0, 2.0])
    A_sub = _relion_euler_angles2matrix(20.0, 35.0, -50.0)

    want = coord_angst / apix + centre - (A_sub @ origin_angst) / apix

    from xmipp_metadata.metadata.relion_tomo import _particle_positions, TiltSeriesGeometry

    geom = TiltSeriesGeometry(
        name="ts", projection=np.eye(4)[None], tomo_size=np.array([4000, 4000, 2000]),
        tilt_image_size=np.array([4000, 4000]), pixel_size=apix)
    df = pd.DataFrame({
        "rlnCenteredCoordinateXAngst": [coord_angst[0]],
        "rlnCenteredCoordinateYAngst": [coord_angst[1]],
        "rlnCenteredCoordinateZAngst": [coord_angst[2]],
        "rlnOriginXAngst": [origin_angst[0]],
        "rlnOriginYAngst": [origin_angst[1]],
        "rlnOriginZAngst": [origin_angst[2]],
        "rlnTomoSubtomogramRot": [20.0],
        "rlnTomoSubtomogramTilt": [35.0],
        "rlnTomoSubtomogramPsi": [-50.0],
    })
    pos, _, _ = _particle_positions(df, geom)
    assert np.allclose(pos[0], want, atol=1e-9)


# --------------------------------------------------------------------------- #
#  2D deformations
# --------------------------------------------------------------------------- #

def test_linear_deformation_matches_relion():
    rng = np.random.default_rng(10)
    image_size = (1024, 768)
    coeffs = rng.normal(0, 0.01, 3)
    model = Linear2DDeformation(image_size, coeffs)

    pts = rng.uniform(0, 1024, size=(50, 2))
    got = model.apply(pts)
    for i, p in enumerate(pts):
        assert np.allclose(got[i], _ref_linear_deformation(p, image_size, coeffs),
                           atol=1e-12)


def test_spline_deformation_matches_relion():
    rng = np.random.default_rng(11)
    image_size = (1024, 768)
    grid = (5, 4)
    coeffs = rng.normal(0, 1.0, 8 * grid[0] * grid[1])
    model = Spline2DDeformation(image_size, grid, coeffs)

    # include points outside the image, which projectPoint clamps into the grid
    pts = rng.uniform(-200, 1200, size=(200, 2))
    got = model.apply(pts)
    for i, p in enumerate(pts):
        assert np.allclose(got[i],
                           _ref_spline_deformation(p, image_size, grid, coeffs),
                           atol=1e-9), i


def test_fourier_deformation_matches_relion():
    rng = np.random.default_rng(12)
    image_size = (1024, 768)
    grid = (6, 6)
    coeffs = rng.normal(0, 1.0, 4 * 64)
    model = Fourier2DDeformation(image_size, grid, coeffs)

    pts = rng.uniform(0, 1024, size=(50, 2))
    got = model.apply(pts)
    for i, p in enumerate(pts):
        assert np.allclose(got[i],
                           _ref_fourier_deformation(p, image_size, grid, coeffs),
                           atol=1e-9), i


def test_zero_deformation_is_identity():
    image_size, grid = (512, 512), (4, 4)
    pts = np.random.default_rng(13).uniform(0, 512, size=(20, 2))
    for model in (Linear2DDeformation(image_size, np.zeros(3)),
                  Spline2DDeformation(image_size, grid, np.zeros(8 * 16)),
                  Fourier2DDeformation(image_size, grid, np.zeros(4 * 64))):
        assert np.allclose(model.apply(pts), pts, atol=1e-12)


def test_spline_needs_a_usable_grid():
    with pytest.raises(ValueError, match="at least 2x2"):
        Spline2DDeformation((512, 512), (1, 4), np.zeros(64))
    with pytest.raises(ValueError, match="needs 128 coefficients"):
        Spline2DDeformation((512, 512), (4, 4), np.zeros(10))


# --------------------------------------------------------------------------- #
#  End to end
# --------------------------------------------------------------------------- #

@pytest.fixture
def tomo_project(tmp_path):
    """A miniature two-tomogram RELION-5 project, with 2D stacks already extracted."""
    rng = np.random.default_rng(5)
    apix, w0, h0, d0 = 1.35, 1024, 1024, 512
    n_frames, n_particles = 11, 7
    box = 64

    tomo_names = ["ts_001", "ts_002"]
    global_rows, tilt_files = [], {}

    for ti, name in enumerate(tomo_names):
        ts = pd.DataFrame({
            "rlnMicrographName": [f"frames/{name}_{f:03d}.mrc" for f in range(n_frames)],
            "rlnTomoXTilt": rng.uniform(-2, 2, n_frames),
            "rlnTomoYTilt": np.linspace(-50, 50, n_frames),
            "rlnTomoZRot": rng.uniform(-180, 180, n_frames),
            "rlnTomoXShiftAngst": rng.uniform(-50, 50, n_frames),
            "rlnTomoYShiftAngst": rng.uniform(-50, 50, n_frames),
            "rlnDefocusU": rng.uniform(10000, 30000, n_frames),
            "rlnDefocusV": rng.uniform(10000, 30000, n_frames),
            "rlnDefocusAngle": rng.uniform(0, 180, n_frames),
            "rlnCtfScalefactor": np.cos(np.deg2rad(np.linspace(-50, 50, n_frames))),
            "rlnMicrographPreExposure": np.arange(n_frames) * 3.0,
        })
        ts_path = tmp_path / f"tilt_series_{name}.star"
        starfile.write({name: ts}, ts_path, overwrite=True)
        tilt_files[name] = ts_path

        global_rows.append({
            "rlnTomoName": name,
            "rlnVoltage": 300.0,
            "rlnSphericalAberration": 2.7,
            "rlnAmplitudeContrast": 0.1,
            "rlnTomoTiltSeriesPixelSize": apix,
            "rlnTomoHand": -1.0,
            "rlnTomoSizeX": w0, "rlnTomoSizeY": h0, "rlnTomoSizeZ": d0,
            "rlnTomoTiltSeriesStarFile": str(ts_path),
        })

    tomograms = tmp_path / "tomograms.star"
    starfile.write({"global": pd.DataFrame(global_rows)}, tomograms, overwrite=True)

    # particles: keep them near the tomogram centre so every frame stays visible
    n = n_particles * len(tomo_names)
    visible = np.ones((n, n_frames), dtype=int)
    particles = pd.DataFrame({
        "rlnTomoName": np.repeat(tomo_names, n_particles),
        "rlnTomoParticleName": [f"{t}/{i}" for t in tomo_names
                                for i in range(n_particles)],
        "rlnCenteredCoordinateXAngst": rng.uniform(-100, 100, n),
        "rlnCenteredCoordinateYAngst": rng.uniform(-100, 100, n),
        "rlnCenteredCoordinateZAngst": rng.uniform(-100, 100, n),
        "rlnOriginXAngst": rng.uniform(-5, 5, n),
        "rlnOriginYAngst": rng.uniform(-5, 5, n),
        "rlnOriginZAngst": rng.uniform(-5, 5, n),
        "rlnAngleRot": rng.uniform(-180, 180, n),
        "rlnAngleTilt": rng.uniform(0, 180, n),
        "rlnAnglePsi": rng.uniform(-180, 180, n),
        "rlnRandomSubset": rng.integers(1, 3, n),
        "rlnImageName": [f"Particles/{i + 1}.mrcs" for i in range(n)],
        "rlnTomoVisibleFrames": ["[" + ",".join(map(str, v)) + "]" for v in visible],
        "rlnOpticsGroup": np.ones(n, dtype=int),
    })
    optics = pd.DataFrame({
        "rlnOpticsGroup": [1],
        "rlnImagePixelSize": [apix * 2.0],
        "rlnImageSize": [box],
    })
    particles_path = tmp_path / "particles.star"
    starfile.write({"optics": optics, "particles": particles},
                   particles_path, overwrite=True)

    return dict(tomograms=tomograms, particles=particles_path, apix=apix,
                n_frames=n_frames, n=n, box=box, w0=w0, h0=h0, d0=d0)


def test_read_tomograms_star(tomo_project):
    geoms = read_tomograms_star(tomo_project["tomograms"],
                                tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))
    assert set(geoms) == {"ts_001", "ts_002"}
    g = geoms["ts_001"]
    assert g.frame_count == tomo_project["n_frames"]
    assert g.pixel_size == tomo_project["apix"]
    assert g.handedness == -1.0
    # every projection matrix must have an orthonormal 3x3 block
    R = g.projection[:, :3, :3]
    assert np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=1e-10)


def test_expansion_shape_and_subtomo_labels(tomo_project):
    df = tomo_star_to_tilt_particles(
        tomo_project["particles"], tomo_project["tomograms"],
        tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))

    n, F = tomo_project["n"], tomo_project["n_frames"]
    assert len(df) == n * F

    labels = df["subtomo_labels"].to_numpy()
    assert labels.dtype.kind in "iu"
    assert labels.min() == 1 and labels.max() == n
    assert np.array_equal(np.unique(labels), np.arange(1, n + 1))
    # every particle contributes exactly one row per tilt
    assert np.all(np.bincount(labels)[1:] == F)

    # 2D stacks: rlnImageName must index the per-particle stack, 1-based, in frame order
    first = df[df["subtomo_labels"] == 1]["rlnImageName"].to_numpy()
    assert [s.split("@")[0] for s in first] == [str(i + 1) for i in range(F)]
    assert all(s.split("@")[1] == "Particles/1.mrcs" for s in first)


def test_expansion_reproduces_relion_composition(tomo_project):
    """The end-to-end angles must equal R_f @ A_sub @ A_part, computed independently."""
    df = tomo_star_to_tilt_particles(
        tomo_project["particles"], tomo_project["tomograms"],
        tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))

    parts = starfile.read(tomo_project["particles"], always_dict=True)["particles"]
    geoms = read_tomograms_star(tomo_project["tomograms"],
                                tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))

    for label in (1, 5, 9, 14):
        rows = df[df["subtomo_labels"] == label].sort_values("rlnTomoFrameIndex")
        p = parts.iloc[label - 1]
        geom = geoms[p["rlnTomoName"]]

        A_part = _relion_euler_angles2matrix(p["rlnAngleRot"], p["rlnAngleTilt"],
                                             p["rlnAnglePsi"])
        for i, (_, r) in enumerate(rows.iterrows()):
            A_tot = geom.projection[i][:3, :3] @ A_part
            got = relion_angles_to_matrix(r["rlnAngleRot"], r["rlnAngleTilt"],
                                          r["rlnAnglePsi"])
            assert np.allclose(got, A_tot, atol=1e-9), (label, i)


def test_defocus_gradient_follows_depth(tomo_project):
    df = tomo_star_to_tilt_particles(
        tomo_project["particles"], tomo_project["tomograms"],
        tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))

    parts = starfile.read(tomo_project["particles"], always_dict=True)["particles"]
    geoms = read_tomograms_star(tomo_project["tomograms"],
                                tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))
    ts = starfile.read(
        starfile.read(tomo_project["tomograms"],
                      always_dict=True)["global"]["rlnTomoTiltSeriesStarFile"][0],
        always_dict=True)["ts_001"]

    from xmipp_metadata.metadata.relion_tomo import _particle_positions

    geom = geoms["ts_001"]
    sub = parts[parts["rlnTomoName"] == "ts_001"].reset_index(drop=True)
    pos, _, _ = _particle_positions(sub, geom)
    projected = geom.project(pos)

    rows = df[df["subtomo_labels"] == 1].sort_values("rlnTomoFrameIndex")
    for i, (_, r) in enumerate(rows.iterrows()):
        z = projected[0, i, 2]
        want = ts["rlnDefocusU"][i] + geom.handedness * geom.pixel_size * 1.0 * z
        assert np.isclose(r["rlnDefocusU"], want, atol=1e-6)
        assert np.isclose(r["rlnDefocusU"] - r["rlnDefocusV"],
                          ts["rlnDefocusU"][i] - ts["rlnDefocusV"][i], atol=1e-6)


def test_shift_modes(tomo_project):
    common = dict(tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))

    zero = tomo_star_to_tilt_particles(tomo_project["particles"],
                                       tomo_project["tomograms"],
                                       shifts="zero", **common)
    assert np.allclose(zero["rlnOriginXAngst"], 0.0)
    assert np.allclose(zero["rlnOriginYAngst"], 0.0)

    residual = tomo_star_to_tilt_particles(tomo_project["particles"],
                                            tomo_project["tomograms"],
                                            shifts="residual", **common)
    apix = tomo_project["apix"]
    # the residual is the sub-pixel part of the crop, so it never exceeds half a pixel
    assert np.all(np.abs(residual["rlnOriginXAngst"]) <= 0.5 * apix + 1e-9)
    assert np.all(np.abs(residual["rlnOriginYAngst"]) <= 0.5 * apix + 1e-9)
    # RELION's relation "true centre = coordinate - origin" must hold exactly, with
    # the true centre recomputed independently from the geometry
    from xmipp_metadata.metadata.relion_tomo import _particle_positions

    geoms = read_tomograms_star(tomo_project["tomograms"], **common)
    parts = starfile.read(tomo_project["particles"], always_dict=True)["particles"]
    for name, geom in geoms.items():
        sub = parts[parts["rlnTomoName"] == name].reset_index(drop=True)
        pos, _, _ = _particle_positions(sub, geom)
        centre = geom.project(pos)[..., :2]                       # (n, F, 2)
        rows = residual[residual["rlnTomoName"] == name]
        got_x = (rows["rlnCoordinateX"].to_numpy()
                 - rows["rlnOriginXAngst"].to_numpy() / apix)
        got_y = (rows["rlnCoordinateY"].to_numpy()
                 - rows["rlnOriginYAngst"].to_numpy() / apix)
        assert np.allclose(got_x, centre[..., 0].ravel(), atol=1e-9)
        assert np.allclose(got_y, centre[..., 1].ravel(), atol=1e-9)

    # The fixture has extracted 2D stacks AND non-zero origins, so "auto" must pick
    # "from_origin": RELION's extraction zeroes the origin (subtomo.cpp writes
    # setParticleOffset(new_id, d3Vector(0,0,0))), so one that is non-zero here can only
    # have been refined afterwards and has to be projected onto each tilt.
    auto = tomo_star_to_tilt_particles(tomo_project["particles"],
                                        tomo_project["tomograms"], **common)
    from_origin = tomo_star_to_tilt_particles(tomo_project["particles"],
                                              tomo_project["tomograms"],
                                              shifts="from_origin", **common)
    pd.testing.assert_frame_equal(auto, from_origin)
    assert not np.allclose(auto["rlnOriginXAngst"], 0.0)


def test_shift_units_pixel(tomo_project):
    common = dict(tilt_image_size=(tomo_project["w0"], tomo_project["h0"]),
                  shifts="residual")
    angst = tomo_star_to_tilt_particles(tomo_project["particles"],
                                        tomo_project["tomograms"], **common)
    px = tomo_star_to_tilt_particles(tomo_project["particles"],
                                     tomo_project["tomograms"],
                                     shift_units="pixel", **common)
    assert "rlnOriginXAngst" in angst.columns and "rlnOriginX" not in angst.columns
    assert "rlnOriginX" in px.columns and "rlnOriginXAngst" not in px.columns
    apix_out = px["rlnImagePixelSize"].to_numpy()
    assert np.allclose(px["rlnOriginX"] * apix_out, angst["rlnOriginXAngst"], atol=1e-9)


def test_xmipp_metadata_reads_tomo_star(tomo_project):
    """The expansion must happen transparently through XmippMetaData."""
    from xmipp_metadata.metadata import XmippMetaData

    md = XmippMetaData(str(tomo_project["particles"]))
    assert md.isTomo
    assert len(md) == tomo_project["n"] * tomo_project["n_frames"]

    labels = md.getMetaDataLabels()
    for expected in ("angleRot", "angleTilt", "anglePsi", "shiftX", "shiftY",
                     "subtomo_labels", "ctfDefocusU", "ctfVoltage", "image"):
        assert expected in labels, expected

    subtomo = md.getMetaDataColumns("subtomo_labels").astype(int)
    assert np.array_equal(np.unique(subtomo), np.arange(1, tomo_project["n"] + 1))

    # shifts must reach the Xmipp table in pixels, which is what consumers expect
    reference = tomo_star_to_tilt_particles(
        tomo_project["particles"], tomo_project["tomograms"],
        shift_units="pixel",
        tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))
    assert np.allclose(md.getMetaDataColumns("shiftX"),
                       reference["rlnOriginX"].to_numpy(), atol=1e-9)
    assert np.allclose(md.getMetaDataColumns("angleRot"),
                       reference["rlnAngleRot"].to_numpy(), atol=1e-9)


def test_xmipp_metadata_tomo_can_be_disabled(tomo_project):
    from xmipp_metadata.metadata import XmippMetaData

    md = XmippMetaData(str(tomo_project["particles"]), tomo=False)
    assert not md.isTomo
    assert len(md) == tomo_project["n"]


def test_deformation_moves_the_crop_but_not_the_pose(tomo_project, tmp_path):
    """
    RELION crops around the deformed position while backprojecting with the
    undeformed 3x3 block, so a deformation must change rlnCoordinateX/Y and leave
    the Euler angles untouched.
    """
    rng = np.random.default_rng(14)
    grid = (4, 4)
    W, H = tomo_project["w0"], tomo_project["h0"]
    F = tomo_project["n_frames"]

    tomograms = starfile.read(tomo_project["tomograms"], always_dict=True)
    g = tomograms["global"].copy()
    g["rlnTomoDeformationGridSizeX"] = grid[0]
    g["rlnTomoDeformationGridSizeY"] = grid[1]
    g["rlnTomoDeformationType"] = "spline"

    coeffs_by_tomo = {}
    for i in range(len(g)):
        name = g["rlnTomoName"][i]
        ts_path = g["rlnTomoTiltSeriesStarFile"][i]
        ts = starfile.read(ts_path, always_dict=True)[name].copy()
        coeffs = rng.normal(0, 2.0, (F, 8 * grid[0] * grid[1]))
        coeffs_by_tomo[name] = coeffs
        ts["rlnTomoDeformationCoefficients"] = [
            "[" + ",".join(f"{v:.8f}" for v in row) + "]" for row in coeffs]
        new_ts = tmp_path / f"deformed_{name}.star"
        starfile.write({name: ts}, new_ts, overwrite=True)
        g.loc[i, "rlnTomoTiltSeriesStarFile"] = str(new_ts)

    deformed_tomograms = tmp_path / "tomograms_deformed.star"
    starfile.write({"global": g}, deformed_tomograms, overwrite=True)

    common = dict(tilt_image_size=(W, H), shifts="residual")
    plain = tomo_star_to_tilt_particles(
        tomo_project["particles"], tomo_project["tomograms"], **common)
    warped = tomo_star_to_tilt_particles(
        tomo_project["particles"], deformed_tomograms, **common)

    # the pose is untouched
    for col in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"):
        assert np.allclose(plain[col], warped[col], atol=1e-9), col
    # ...and so is the depth-driven defocus
    assert np.allclose(plain["rlnDefocusU"], warped["rlnDefocusU"], atol=1e-9)
    # but the crop centre moved
    assert not np.allclose(plain["rlnCoordinateX"], warped["rlnCoordinateX"])

    # and it moved by exactly the model's displacement
    geoms = read_tomograms_star(deformed_tomograms, tilt_image_size=(W, H))
    geom = geoms["ts_001"]
    assert geom.has_deformations
    model = Spline2DDeformation((W, H), grid, coeffs_by_tomo["ts_001"][0])

    from xmipp_metadata.metadata.relion_tomo import _particle_positions

    parts = starfile.read(tomo_project["particles"], always_dict=True)["particles"]
    sub = parts[parts["rlnTomoName"] == "ts_001"].reset_index(drop=True)
    pos, _, _ = _particle_positions(sub, geom)
    undeformed = (np.einsum('ij,nj->ni', geom.projection[0, :3, :3], pos)
                  + geom.projection[0, :3, 3])
    assert np.allclose(geom.project(pos)[:, 0, :2], model.apply(undeformed[:, :2]),
                       atol=1e-9)


def test_trajectories_shift_the_crop_centre(tomo_project, tmp_path):
    rng = np.random.default_rng(15)
    W, H = tomo_project["w0"], tomo_project["h0"]
    F, n = tomo_project["n_frames"], tomo_project["n"]
    apix = tomo_project["apix"]

    parts = starfile.read(tomo_project["particles"], always_dict=True)["particles"]
    names = list(parts["rlnTomoParticleName"])
    shifts = {name: rng.normal(0, 8.0, (F, 3)) for name in names}

    motion = {name: pd.DataFrame({"rlnOriginXAngst": s[:, 0],
                                  "rlnOriginYAngst": s[:, 1],
                                  "rlnOriginZAngst": s[:, 2]})
              for name, s in shifts.items()}
    motion_path = tmp_path / "motion.star"
    starfile.write(motion, motion_path, overwrite=True)

    loaded = read_trajectories_star(motion_path)
    assert set(loaded) == set(names)
    assert np.allclose(loaded[names[0]], shifts[names[0]], atol=1e-6)

    common = dict(tilt_image_size=(W, H), shifts="residual")
    plain = tomo_star_to_tilt_particles(
        tomo_project["particles"], tomo_project["tomograms"], **common)
    moved = tomo_star_to_tilt_particles(
        tomo_project["particles"], tomo_project["tomograms"],
        trajectories=str(motion_path), **common)

    assert len(plain) == len(moved) == n * F
    # RELION adds the trajectory to the position; the pose is unchanged
    for col in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"):
        assert np.allclose(plain[col], moved[col], atol=1e-9), col
    assert not np.allclose(plain["rlnCoordinateX"], moved["rlnCoordinateX"])

    # verify the displacement for one particle against a direct computation
    from xmipp_metadata.metadata.relion_tomo import _particle_positions

    geom = read_tomograms_star(tomo_project["tomograms"],
                               tilt_image_size=(W, H))["ts_001"]
    sub = parts[parts["rlnTomoName"] == "ts_001"].reset_index(drop=True)
    pos, _, _ = _particle_positions(sub, geom)
    traj = np.stack([shifts[nm] for nm in sub["rlnTomoParticleName"]]) / apix
    want = geom.project(pos[:, None, :] + traj)
    rows = moved[moved["rlnTomoName"] == "ts_001"]
    assert np.allclose(rows["rlnCoordinateX"].to_numpy(),
                       np.rint(want[..., 0]).ravel(), atol=1e-9)


def test_trajectories_need_particle_names(tomo_project, tmp_path):
    parts = starfile.read(tomo_project["particles"], always_dict=True)
    parts["particles"] = parts["particles"].drop(columns=["rlnTomoParticleName"])
    path = tmp_path / "particles_noname.star"
    starfile.write(parts, path, overwrite=True)

    with pytest.raises(ValueError, match="rlnTomoParticleName"):
        tomo_star_to_tilt_particles(
            path, tomo_project["tomograms"],
            tilt_image_size=(tomo_project["w0"], tomo_project["h0"]),
            trajectories={"a": np.zeros((tomo_project["n_frames"], 3))})


def _write_tomograms_with_matrices(tmp_path, tomo_project, name, zero_translation):
    """Rewrite the fixture's geometry as explicit rlnTomoProj* matrices."""
    geoms = read_tomograms_star(tomo_project["tomograms"],
                                tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))
    g = starfile.read(tomo_project["tomograms"], always_dict=True)["global"].copy()
    blocks = {}
    for i in range(len(g)):
        tomo = g["rlnTomoName"][i]
        proj = np.array(geoms[tomo].projection, copy=True)
        if zero_translation:
            proj[:, :3, 3] = 0.0
        ts = starfile.read(g["rlnTomoTiltSeriesStarFile"][i], always_dict=True)[tomo].copy()
        for row, label in enumerate(("rlnTomoProjX", "rlnTomoProjY",
                                     "rlnTomoProjZ", "rlnTomoProjW")):
            ts[label] = ["[" + ",".join(f"{v:.10f}" for v in proj[f, row]) + "]"
                         for f in range(proj.shape[0])]
        blocks[tomo] = ts
    g = g.drop(columns=["rlnTomoTiltSeriesStarFile"])
    path = tmp_path / name
    starfile.write({"global": g, **blocks}, path, overwrite=True)
    return path


def test_from_origin_matches_a_direct_projection(tomo_project):
    """
    The refined 3D origin, projected onto each tilt. Checked against a direct
    computation rather than against the converter's own intermediates.
    """
    W, H = tomo_project["w0"], tomo_project["h0"]
    apix = tomo_project["apix"]
    df = tomo_star_to_tilt_particles(
        tomo_project["particles"], tomo_project["tomograms"],
        shifts="from_origin", tilt_image_size=(W, H))

    geoms = read_tomograms_star(tomo_project["tomograms"], tilt_image_size=(W, H))
    parts = starfile.read(tomo_project["particles"], always_dict=True)["particles"]

    for name, geom in geoms.items():
        sub = parts[parts["rlnTomoName"] == name].reset_index(drop=True)
        offset = np.stack([sub["rlnOriginXAngst"].to_numpy(),
                           sub["rlnOriginYAngst"].to_numpy(),
                           sub["rlnOriginZAngst"].to_numpy()], axis=-1)
        # the fixture has no rlnTomoSubtomogram* angles, so A_sub is the identity
        delta = -offset / apix
        delta_2d = np.einsum('fij,nj->nfi', geom.projection[:, :3, :3], delta)[..., :2]
        want = -delta_2d * apix

        rows = df[df["rlnTomoName"] == name]
        assert np.allclose(rows["rlnOriginXAngst"].to_numpy(), want[..., 0].ravel(),
                           atol=1e-9)
        assert np.allclose(rows["rlnOriginYAngst"].to_numpy(), want[..., 1].ravel(),
                           atol=1e-9)

    # a real refinement moves the particle, so this must not be a no-op
    assert np.abs(df["rlnOriginXAngst"]).max() > 1e-3


def test_from_origin_is_immune_to_a_missing_translation(tomo_project, tmp_path):
    """
    The justification for the whole mode: it projects a *difference* of positions, so the
    translation column cancels and WarpTools' rotation-only matrices give the same answer.
    """
    W, H = tomo_project["w0"], tomo_project["h0"]
    full = _write_tomograms_with_matrices(tmp_path, tomo_project, "tg_full.star", False)
    rot_only = _write_tomograms_with_matrices(tmp_path, tomo_project, "tg_rot.star", True)

    assert not read_tomograms_star(full)["ts_001"].is_rotation_only
    assert read_tomograms_star(rot_only)["ts_001"].is_rotation_only

    a = tomo_star_to_tilt_particles(tomo_project["particles"], full,
                                    shifts="from_origin", tilt_image_size=(W, H))
    b = tomo_star_to_tilt_particles(tomo_project["particles"], rot_only,
                                    shifts="from_origin", tilt_image_size=(W, H))

    for col in ("rlnOriginXAngst", "rlnOriginYAngst",
                "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnDefocusU"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy(), atol=1e-9), col

    # ...whereas the absolute crop coordinate is exactly what does NOT survive
    assert not np.allclose(a["rlnCoordinateX"].to_numpy(),
                           b["rlnCoordinateX"].to_numpy())


def test_from_origin_recovers_the_refined_centre(tomo_project):
    """coordinate - origin/apix must land on the projection of the *refined* position."""
    W, H = tomo_project["w0"], tomo_project["h0"]
    apix = tomo_project["apix"]
    df = tomo_star_to_tilt_particles(
        tomo_project["particles"], tomo_project["tomograms"],
        shifts="from_origin", tilt_image_size=(W, H))

    from xmipp_metadata.metadata.relion_tomo import _particle_positions

    geoms = read_tomograms_star(tomo_project["tomograms"], tilt_image_size=(W, H))
    parts = starfile.read(tomo_project["particles"], always_dict=True)["particles"]

    for name, geom in geoms.items():
        sub = parts[parts["rlnTomoName"] == name].reset_index(drop=True)
        pos, _, delta = _particle_positions(sub, geom)
        refined = geom.project(pos)[..., :2]              # where the particle really is
        extraction = geom.project(pos - delta)[..., :2]   # where the box was cut

        rows = df[df["rlnTomoName"] == name]
        got = (np.stack([rows["rlnCoordinateX"].to_numpy(),
                         rows["rlnCoordinateY"].to_numpy()], axis=-1)
               - np.stack([rows["rlnOriginXAngst"].to_numpy(),
                           rows["rlnOriginYAngst"].to_numpy()], axis=-1) / apix)
        # RELION's relation must hold exactly: coordinate - origin is the refined centre
        assert np.allclose(got, refined.reshape(-1, 2), atol=1e-9)
        # ...and the coordinate itself is the centre the box was actually cut at
        assert np.allclose(
            np.stack([rows["rlnCoordinateX"].to_numpy(),
                      rows["rlnCoordinateY"].to_numpy()], axis=-1),
            extraction.reshape(-1, 2), atol=1e-9)


def test_from_origin_is_a_no_op_without_origins(tomo_project, tmp_path):
    blocks = starfile.read(tomo_project["particles"], always_dict=True)
    for c in ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"):
        blocks["particles"][c] = 0.0
    path = tmp_path / "particles_nozero.star"
    starfile.write(blocks, path, overwrite=True)

    df = tomo_star_to_tilt_particles(
        path, tomo_project["tomograms"], shifts="from_origin",
        tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))
    assert np.allclose(df["rlnOriginXAngst"], 0.0)
    assert np.allclose(df["rlnOriginYAngst"], 0.0)


def test_dropping_refined_origins_is_flagged(tomo_project):
    """Asking for shifts='zero' explicitly still throws the refinement away -- say so."""
    with pytest.warns(RuntimeWarning, match="shifts='from_origin'"):
        tomo_star_to_tilt_particles(
            tomo_project["particles"], tomo_project["tomograms"], shifts="zero",
            tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))


def test_auto_does_not_drop_refined_origins(tomo_project, recwarn):
    """The default must not silently discard a refinement -- that was the old behaviour."""
    df = tomo_star_to_tilt_particles(
        tomo_project["particles"], tomo_project["tomograms"],
        tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))
    assert np.abs(df["rlnOriginXAngst"].to_numpy()).max() > 0
    assert not [w for w in recwarn if "discarded" in str(w.message)]


def test_auto_picks_zero_when_there_is_nothing_to_apply(tomo_project, tmp_path):
    """With the origins genuinely zero, auto must not invent a shift."""
    parts = starfile.read(tomo_project["particles"], always_dict=True)
    for c in ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"):
        parts["particles"][c] = 0.0
    path = tmp_path / "particles_no_origin.star"
    starfile.write(parts, path, overwrite=True)

    df = tomo_star_to_tilt_particles(
        path, tomo_project["tomograms"],
        tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))
    assert np.allclose(df["rlnOriginXAngst"], 0.0)
    assert np.allclose(df["rlnOriginYAngst"], 0.0)


def test_visibility_is_honoured(tomo_project, tmp_path):
    """Frames flagged invisible must not produce rows, and stack indices must renumber."""
    parts = starfile.read(tomo_project["particles"], always_dict=True)
    F = tomo_project["n_frames"]
    visible = np.ones((tomo_project["n"], F), dtype=int)
    visible[:, 0] = 0
    visible[:, 3] = 0
    parts["particles"]["rlnTomoVisibleFrames"] = [
        "[" + ",".join(map(str, v)) + "]" for v in visible]
    path = tmp_path / "particles_masked.star"
    starfile.write(parts, path, overwrite=True)

    df = tomo_star_to_tilt_particles(
        path, tomo_project["tomograms"],
        tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))

    assert len(df) == tomo_project["n"] * (F - 2)
    rows = df[df["subtomo_labels"] == 1].sort_values("rlnTomoFrameIndex")
    assert list(rows["rlnTomoFrameIndex"]) == [2, 3, 5, 6, 7, 8, 9, 10, 11]
    assert [s.split("@")[0] for s in rows["rlnImageName"]] == \
           [str(i + 1) for i in range(F - 2)]


# --------------------------------------------------------------------------- #
#  Optimisation sets
# --------------------------------------------------------------------------- #

def _write_optimisation_set(path, block_name, particles, tomograms, extra=None):
    """
    Write a RELION optimisation set by hand.

    It is written as text rather than through ``starfile.write`` so the block name
    can be controlled exactly -- the point of these tests is that a loop-less block
    must be found by its labels, not by what the block happens to be called.
    """
    rows = {"rlnTomoParticlesFile": str(particles),
            "rlnTomoTomogramsFile": str(tomograms)}
    rows.update(extra or {})
    lines = ["# version 50001", "", f"data_{block_name}", ""]
    lines += [f"_{k}{' ' * 8}{v}" for k, v in rows.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("block_name", ["optimisation_set", ""])
def test_optimisation_set_resolves_its_files(tomo_project, tmp_path, block_name):
    """
    starfile hands a loop-less block back as a plain dict, not a frame. If that is
    dropped the optimisation set vanishes and the file reads as an ordinary
    single-particle STAR, which is silently wrong rather than an error.
    """
    opt = _write_optimisation_set(
        tmp_path / "run_optimisation_set.star", block_name,
        tomo_project["particles"], tomo_project["tomograms"])

    resolved = read_optimisation_set(opt)
    assert resolved["particles"] == str(tomo_project["particles"])
    assert resolved["tomograms"] == str(tomo_project["tomograms"])
    assert resolved["trajectories"] is None

    assert is_relion_tomo_star(_read_star(opt))


@pytest.mark.parametrize("block_name", ["optimisation_set", ""])
def test_expansion_through_an_optimisation_set(tomo_project, tmp_path, block_name):
    """Going in through the optimisation set must give the same table as going direct."""
    opt = _write_optimisation_set(
        tmp_path / "run_optimisation_set.star", block_name,
        tomo_project["particles"], tomo_project["tomograms"])

    kwargs = dict(tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))
    direct = tomo_star_to_tilt_particles(
        tomo_project["particles"], tomo_project["tomograms"], **kwargs)
    # the tomograms STAR is deliberately not passed: it has to come from the set
    through = tomo_star_to_tilt_particles(opt, **kwargs)

    pd.testing.assert_frame_equal(direct, through)


def test_optimisation_set_reaches_xmipp_metadata(tomo_project, tmp_path):
    """The header sniff must route an optimisation set to the tomo expansion."""
    opt = _write_optimisation_set(
        tmp_path / "run_optimisation_set.star", "optimisation_set",
        tomo_project["particles"], tomo_project["tomograms"])

    assert XmippMetaData._sniffTomoKind(str(opt)) == "relion"
    assert XmippMetaData._sniffTomoKind(str(tomo_project["particles"])) == "relion"

    md = XmippMetaData(str(opt))
    assert md.isTomo and md.tomoFormat == "relion"
    assert len(md) == tomo_project["n"] * tomo_project["n_frames"]
    subtomo = md.getMetaDataColumns("subtomo_labels").astype(int)
    assert np.array_equal(np.unique(subtomo), np.arange(1, tomo_project["n"] + 1))


def test_a_file_that_is_not_an_optimisation_set_is_rejected(tomo_project):
    with pytest.raises(ValueError, match="rlnTomoParticlesFile"):
        read_optimisation_set(tomo_project["tomograms"])


def test_loop_less_blocks_survive_a_plain_star_read(tmp_path):
    """
    A non-tomography STAR with a loop-less block used to crash the label converter
    with AttributeError: 'dict' object has no attribute 'columns'.
    """
    path = tmp_path / "mixed.star"
    parts = pd.DataFrame({"rlnImageName": ["1@a.mrcs", "2@a.mrcs"],
                          "rlnAngleRot": [10.0, 20.0]})
    starfile.write({"general": pd.Series({"rlnNrParticles": 2}),
                    "particles": parts}, path, overwrite=True)

    md = XmippMetaData(str(path))
    assert not md.isTomo
    assert len(md) == 2


def test_subtomogram_matrix_multiplies_on_the_left(tomo_project, tmp_path):
    """
    RELION composes ``A_subtomogram * A_particle`` (ParticleSet::getMatrix3x3), so the
    order matters as soon as the subtomogram orientation is not the identity -- which
    is exactly the case for particles extracted by Warp and then refined in RELION.
    The main fixture carries no rlnTomoSubtomogram*, so it cannot tell the two orders
    apart; this test adds them and pins the order.
    """
    rng = np.random.default_rng(11)
    parts = starfile.read(tomo_project["particles"], always_dict=True)
    n = len(parts["particles"])
    sub = {"rlnTomoSubtomogramRot": rng.uniform(-180, 180, n),
           "rlnTomoSubtomogramTilt": rng.uniform(0, 180, n),
           "rlnTomoSubtomogramPsi": rng.uniform(-180, 180, n)}
    for k, v in sub.items():
        parts["particles"][k] = v
    path = tmp_path / "particles_with_subtomo.star"
    starfile.write(parts, path, overwrite=True)

    df = tomo_star_to_tilt_particles(
        path, tomo_project["tomograms"],
        tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))

    geoms = read_tomograms_star(tomo_project["tomograms"],
                                tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))
    wrong_order_seen = False
    for label in (1, 6, 14):
        p = parts["particles"].iloc[label - 1]
        geom = geoms[p["rlnTomoName"]]
        A_part = _relion_euler_angles2matrix(p["rlnAngleRot"], p["rlnAngleTilt"],
                                             p["rlnAnglePsi"])
        A_sub = _relion_euler_angles2matrix(p["rlnTomoSubtomogramRot"],
                                            p["rlnTomoSubtomogramTilt"],
                                            p["rlnTomoSubtomogramPsi"])
        rows = df[df["subtomo_labels"] == label].sort_values("rlnTomoFrameIndex")
        for i, (_, r) in enumerate(rows.iterrows()):
            got = relion_angles_to_matrix(r["rlnAngleRot"], r["rlnAngleTilt"],
                                          r["rlnAnglePsi"])
            assert np.allclose(got, geom.projection[i][:3, :3] @ A_sub @ A_part,
                               atol=1e-9), (label, i)
            # and the reversed composition must be genuinely different, or the test
            # would pass for the wrong reason
            if not np.allclose(A_sub @ A_part, A_part @ A_sub, atol=1e-6):
                wrong_order_seen = True
    assert wrong_order_seen, "the fixture failed to make the two orders distinguishable"


# --------------------------------------------------------------------------- #
#  Project-relative image paths
# --------------------------------------------------------------------------- #

@pytest.fixture
def relion_project_layout(tomo_project, tmp_path):
    """A RELION-shaped project: the star file two job directories below the root, with
    rlnImageName written relative to the root rather than to the star file."""
    import mrcfile

    root = tmp_path / "project"
    (root / "Refine3D" / "job012").mkdir(parents=True)
    (root / "Extract" / "job010" / "Particles").mkdir(parents=True)

    parts = starfile.read(tomo_project["particles"], always_dict=True)
    n, F = tomo_project["n"], tomo_project["n_frames"]
    names = []
    for i in range(n):
        rel = f"Extract/job010/Particles/{i + 1}.mrcs"
        with mrcfile.new(root / rel, overwrite=True) as m:
            m.set_data(np.zeros((F, tomo_project["box"], tomo_project["box"]), np.float32))
        names.append(rel)
    parts["particles"]["rlnImageName"] = names

    particles = root / "Refine3D" / "job012" / "run_data.star"
    starfile.write(parts, particles, overwrite=True)
    opt = _write_optimisation_set(root / "Refine3D" / "job012" / "run_optimisation_set.star",
                                  "optimisation_set", particles, tomo_project["tomograms"])
    return dict(root=root, optimisation_set=opt, n=n, n_frames=F)


def test_image_paths_resolve_from_any_working_directory(relion_project_layout, monkeypatch,
                                                        tmp_path):
    """
    RELION anchors rlnImageName at the project root, so 'Extract/job010/Particles/1.mrcs'
    in a star file living in Refine3D/job012/ is relative to neither the star file's own
    directory nor whatever directory the program happens to be run from.
    """
    layout = relion_project_layout
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()

    for cwd in (layout["root"], elsewhere, tmp_path):
        monkeypatch.chdir(cwd)
        md = XmippMetaData(str(layout["optimisation_set"]))
        assert md.binaries, f"stacks not found with cwd={cwd}"
        assert md.getMetaDataImage(0).shape[-1] == 64
        assert len(md) == layout["n"] * layout["n_frames"]


def test_missing_stacks_say_why(relion_project_layout, monkeypatch, tmp_path):
    """A genuinely absent stack must still report cleanly, and name the convention."""
    layout = relion_project_layout
    for f in (layout["root"] / "Extract" / "job010" / "Particles").glob("*.mrcs"):
        f.unlink()
    monkeypatch.chdir(tmp_path)

    with pytest.warns(RuntimeWarning, match="project root"):
        md = XmippMetaData(str(layout["optimisation_set"]))
    assert not md.binaries


def test_write_rebases_project_relative_paths(relion_project_layout, monkeypatch, tmp_path):
    """updateImagePaths must re-point the paths from the project root to the new file."""
    layout = relion_project_layout
    monkeypatch.chdir(tmp_path)
    md = XmippMetaData(str(layout["optimisation_set"]))

    out = tmp_path / "out" / "tilts.xmd"
    out.parent.mkdir()
    md.write(str(out), updateImagePaths=True)

    monkeypatch.chdir(out.parent)
    again = XmippMetaData(str(out))
    assert again.binaries, "the rewritten paths do not resolve from the new location"
    assert again.getMetaDataImage(0).shape[-1] == 64


def test_data_general_block_does_not_derail_the_read(tomo_project, tmp_path):
    """
    RELION-5's Extract writes a loop-less ``data_general`` block carrying
    ``rlnTomoSubTomosAre2DStacks``. It has to be read past, not tripped over -- and it is
    written by hand here because starfile 0.5.13 silently drops a loop-less block on
    write, so a fixture built with starfile alone would not test anything.
    """
    parts = starfile.read(tomo_project["particles"], always_dict=True)
    path = tmp_path / "particles_general.star"
    starfile.write(parts, path, overwrite=True)
    path.write_text("# version 50001\n\ndata_general\n\n"
                    "_rlnTomoSubTomosAre2DStacks            1\n\n\n" + path.read_text())

    reference = tomo_star_to_tilt_particles(
        tomo_project["particles"], tomo_project["tomograms"],
        tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))
    got = tomo_star_to_tilt_particles(
        path, tomo_project["tomograms"],
        tilt_image_size=(tomo_project["w0"], tomo_project["h0"]))
    pd.testing.assert_frame_equal(reference, got)
