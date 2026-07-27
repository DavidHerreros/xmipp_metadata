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
RELION tomography -> per-tilt-image (single-particle-like) metadata.

A RELION-5 subtomogram data set stores the alignment of a particle in *two*
separate places:

  * the tilt-series geometry, one 4x4 projection matrix per tilt image, which
    maps a point of the tomogram onto a pixel of that tilt image.  It lives in
    the tomograms STAR file (``rlnTomoProjX/Y/Z/W``, or the
    ``rlnTomoXTilt/YTilt/ZRot/XShiftAngst/YShiftAngst`` parameterisation).
  * the subtomogram alignment, one 3D orientation + one 3D shift per particle,
    which lives in the particles STAR file (``rlnAngleRot/Tilt/Psi`` and
    ``rlnOriginX/Y/ZAngst``).

Neither one on its own is a usable 2D pose.  This module composes them into the
per (particle, tilt image) pose that a *single-particle* reconstruction needs, so
that back-projecting the 2D tilt images with the resulting Euler angles and
shifts reproduces the very volume ``relion_tomo_reconstruct_particle`` produces.

Everything here is derived from RELION's own source, not from a re-derivation:

  ``Tomogram::setProjectionMatrix``              -> :func:`projection_matrix_from_angles`
  ``ParticleSet::getPosition``                   -> :func:`ParticleGeometry.positions`
  ``ParticleSet::getMatrix3x3``                  -> ``A_subtomogram @ A_particle``
  ``ParticleSet::getMatrix4x4``                  -> ``Ts * R * Tc``
  ``TomoExtraction::extractAt2D_Fourier``        -> the shift convention
  ``reconstruct_particle.cpp``                   -> ``projPart = projCut * particleToTomo``
  ``Tomogram::getCtf`` / ``getDepthOffset``      -> :func:`TiltSeriesGeometry.defocus_offset`
  ``Euler_angles2matrix`` / ``Euler_matrix2angles`` -> :func:`relion_angles_to_matrix` and inverse

The key algebraic result (see ``docs`` in :func:`tomo_star_to_tilt_particles`) is
that once the tilt image is cropped around the projected particle centre, the
composed transform is a *pure rotation about the box centre*::

    projPart_f(u) = R_f @ A_sub @ A_part @ (u - s/2) + s/2

so the per-tilt-image RELION Euler triplet is simply the decomposition of
``R_f @ A_sub @ A_part``, and the only residual translation is the sub-pixel
part of the crop.
"""

import os
import re
import warnings
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
import starfile


__all__ = [
    "relion_angles_to_matrix",
    "matrix_to_relion_angles",
    "projection_matrix_from_angles",
    "Linear2DDeformation",
    "Spline2DDeformation",
    "Fourier2DDeformation",
    "build_deformation",
    "TiltSeriesGeometry",
    "read_tomograms_star",
    "read_optimisation_set",
    "read_trajectories_star",
    "is_relion_tomo_star",
    "tomo_star_to_tilt_particles",
]


_FLT_EPSILON = float(np.finfo(np.float32).eps)


# --------------------------------------------------------------------------- #
#  Euler conventions
# --------------------------------------------------------------------------- #

def relion_angles_to_matrix(rot, tilt, psi):
    """
    RELION's ``Euler_angles2matrix`` (a.k.a. ``Euler::anglesToMatrix3``), vectorised.

    The returned matrix ``A`` maps *3D volume coordinates onto 2D image
    coordinates*: the first two rows of ``A`` are the in-plane axes of the
    projection, the third row is the projection (viewing) direction.  Equivalently
    ``A = (Rz(rot) @ Ry(tilt) @ Rz(psi)).T`` with right-handed active rotations.

        :param rot, tilt, psi --> Euler angles in DEGREES (scalars or arrays)
        :returns: (..., 3, 3) array of rotation matrices
    """
    rot = np.deg2rad(np.asarray(rot, dtype=np.float64))
    tilt = np.deg2rad(np.asarray(tilt, dtype=np.float64))
    psi = np.deg2rad(np.asarray(psi, dtype=np.float64))
    rot, tilt, psi = np.broadcast_arrays(rot, tilt, psi)

    ca, sa = np.cos(rot), np.sin(rot)
    cb, sb = np.cos(tilt), np.sin(tilt)
    cg, sg = np.cos(psi), np.sin(psi)

    cc, cs = cb * ca, cb * sa
    sc, ss = sb * ca, sb * sa

    A = np.empty(rot.shape + (3, 3), dtype=np.float64)
    A[..., 0, 0] = cg * cc - sg * sa
    A[..., 0, 1] = cg * cs + sg * ca
    A[..., 0, 2] = -cg * sb
    A[..., 1, 0] = -sg * cc - cg * sa
    A[..., 1, 1] = -sg * cs + cg * ca
    A[..., 1, 2] = sg * sb
    A[..., 2, 0] = sc
    A[..., 2, 1] = ss
    A[..., 2, 2] = cb
    return A


def matrix_to_relion_angles(A):
    """
    RELION's ``Euler_matrix2angles``, vectorised, including both gimbal-lock
    branches.  Exact inverse of :func:`relion_angles_to_matrix`.

        :param A --> (..., 3, 3) rotation matrices
        :returns: (rot, tilt, psi) in DEGREES, each of shape A.shape[:-2]
    """
    A = np.asarray(A, dtype=np.float64)

    a02, a12, a22 = A[..., 0, 2], A[..., 1, 2], A[..., 2, 2]
    a21, a20 = A[..., 2, 1], A[..., 2, 0]
    a10, a00 = A[..., 1, 0], A[..., 0, 0]

    abs_sb = np.sqrt(a02 * a02 + a12 * a12)

    # ---- regular branch -------------------------------------------------- #
    # np.errstate guards the degenerate rows, which are discarded below anyway.
    with np.errstate(invalid="ignore", divide="ignore"):
        psi = np.arctan2(a12, -a02)
        rot = np.arctan2(a21, a20)

        sin_psi = np.sin(psi)
        # RELION's SGN() maps 0 -> +1, so np.sign() is not a drop-in replacement
        def _sgn(x):
            return np.where(x >= 0, 1.0, -1.0)

        sign_sb = np.where(
            np.abs(sin_psi) < _FLT_EPSILON,
            _sgn(-a02 / np.cos(psi)),
            np.where(sin_psi > 0, _sgn(a12), -_sgn(a12)),
        )
        tilt = np.arctan2(sign_sb * abs_sb, a22)

    # ---- gimbal lock (tilt == 0 or 180) ---------------------------------- #
    up = a22 >= 0
    rot_d = np.zeros_like(abs_sb)
    tilt_d = np.where(up, 0.0, np.pi)
    psi_d = np.where(up, np.arctan2(-a10, a00), np.arctan2(a10, -a00))

    regular = abs_sb > 16.0 * _FLT_EPSILON
    rot = np.where(regular, rot, rot_d)
    tilt = np.where(regular, tilt, tilt_d)
    psi = np.where(regular, psi, psi_d)

    return np.rad2deg(rot), np.rad2deg(tilt), np.rad2deg(psi)


def _axis_rotation(axis, angle_deg):
    """
    Rodrigues rotation matching gravis ``t3Matrix::rotation(axis, angle)``, which
    RELION's ``setProjectionMatrix`` calls.  Note the angle is in DEGREES there.

        :param axis --> unit axis, shape (3,)
        :param angle_deg --> angle(s) in degrees, shape (...,)
        :returns: (..., 3, 3)
    """
    n = np.asarray(axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    a = np.deg2rad(np.asarray(angle_deg, dtype=np.float64))

    S = np.array([[0.0, -n[2], n[1]],
                  [n[2], 0.0, -n[0]],
                  [-n[1], n[0], 0.0]])
    nnt = np.outer(n, n)
    I = np.eye(3)

    c = np.cos(a)[..., None, None]
    s = np.sin(a)[..., None, None]
    return nnt + c * (I - nnt) + s * S


# --------------------------------------------------------------------------- #
#  2D image deformations
# --------------------------------------------------------------------------- #
#
# `relion_tomo_align --deformation` fits a 2D warp per tilt image on top of the
# linear tilt-series geometry. RELION applies it in ``Tomogram::projectPoint``,
# *after* the projection matrix, and every model is of the form
#
#     apply(pl) = pl + computeShift(pl, coefficients)
#
# Crucially it never enters the pose: ``extractAt2D_Fourier`` crops around the
# deformed centre while ``projCut`` keeps the *undeformed* matrix, and
# ``FourierBackprojection::backprojectSlice_backward`` reads only the 3x3 block
# and discards the translation column. So a deformation moves *where the tilt
# image is cropped* and nothing else.

class Linear2DDeformation:
    """RELION ``Linear2DDeformationModel``: 3 coefficients, an affine shear."""

    n_coefficients = 3

    def __init__(self, image_size, coefficients):
        self.centre = 0.5 * np.asarray(image_size, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64).ravel()[:3]

    def apply(self, xy):
        axx, axy, ayy = self.coefficients
        r = np.asarray(xy, dtype=np.float64) - self.centre
        return xy + np.stack([axx * r[..., 0],
                              axy * r[..., 0] + ayy * r[..., 1]], axis=-1)


class Spline2DDeformation:
    """
    RELION ``Spline2DDeformationModel``: a bicubic Hermite spline on a
    ``grid_x * grid_y`` lattice, 8 coefficients per node (4 per output dimension:
    value, slope_x, slope_y, twist).
    """

    def __init__(self, image_size, grid_size, coefficients):
        self.image_size = np.asarray(image_size, dtype=np.float64)
        self.grid_size = np.asarray(grid_size, dtype=np.int64)
        if np.any(self.grid_size < 2):
            raise ValueError(f"spline deformation needs a grid of at least 2x2, "
                             f"got {tuple(self.grid_size)}")
        self.grid_spacing = self.image_size / (self.grid_size - 1.0)

        gx, gy = int(self.grid_size[0]), int(self.grid_size[1])
        n = 8 * gx * gy
        coefficients = np.asarray(coefficients, dtype=np.float64).ravel()
        if coefficients.size < n:
            raise ValueError(f"spline deformation needs {n} coefficients, "
                             f"got {coefficients.size}")
        # RawImage<DataPoint>(gx, gy, 2) with DataPoint = 4 doubles, so the flat
        # index of component c of node (x, y) of dimension d is 4*(x + y*gx + d*gx*gy) + c
        self.nodes = coefficients[:n].reshape(2, gy, gx, 4)

    @staticmethod
    def _hermite(t):
        t2, t3 = t * t, t * t * t
        return np.stack([1.0 - 3.0 * t2 + 2.0 * t3,
                         3.0 * t2 - 2.0 * t3,
                         t - 2.0 * t2 + t3,
                         -t2 + t3], axis=-1)

    def apply(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        eps = 1e-10
        g = xy / self.grid_spacing
        g = np.clip(g, 0.0, self.grid_size - 1 - eps)
        cell = g.astype(np.int64)
        frc = g - cell

        vx = self._hermite(frc[..., 0])
        vy = self._hermite(frc[..., 1])
        cx, cy = cell[..., 0], cell[..., 1]

        VALUE, SLOPE_X, SLOPE_Y, TWIST = 0, 1, 2, 3
        shift = np.empty(xy.shape[:-1] + (2,), dtype=np.float64)
        for dim in range(2):
            node = self.nodes[dim]
            d00, d01 = node[cy, cx], node[cy + 1, cx]
            d10, d11 = node[cy, cx + 1], node[cy + 1, cx + 1]

            # gravis d4Matrix is row-major; this is RELION's F, laid out verbatim
            F = np.stack([
                np.stack([d00[..., VALUE], d01[..., VALUE],
                          d00[..., SLOPE_Y], d01[..., SLOPE_Y]], axis=-1),
                np.stack([d10[..., VALUE], d11[..., VALUE],
                          d10[..., SLOPE_Y], d11[..., SLOPE_Y]], axis=-1),
                np.stack([d00[..., SLOPE_X], d01[..., SLOPE_X],
                          d00[..., TWIST], d01[..., TWIST]], axis=-1),
                np.stack([d10[..., SLOPE_X], d11[..., SLOPE_X],
                          d10[..., TWIST], d11[..., TWIST]], axis=-1),
            ], axis=-2)
            shift[..., dim] = np.einsum('...i,...ij,...j->...', vx, F, vy)

        return xy + shift


class Fourier2DDeformation:
    """
    RELION ``Fourier2DDeformationModel``: a truncated 2D Fourier series, with one
    complex coefficient per (spatial frequency, output dimension).

    Note RELION's own two views of this coefficient block disagree on their stride
    (the wrapper allocates ``(gx/2+1) * gy * 2`` complex values while
    ``computeShift`` indexes with a stride of ``len(spatialFrequencies)``). What is
    reproduced here is ``computeShift``, since that is the code that is evaluated.
    """

    def __init__(self, image_size, grid_size, coefficients):
        self.image_size = np.asarray(image_size, dtype=np.float64)
        gx, gy = int(grid_size[0]), int(grid_size[1])
        self.grid_size = (gx, gy)

        freqs = []
        for y in range(gy):
            if y < gy // 2:
                for x in range(1 if y == 0 else 0, gx // 2 + 1):
                    freqs.append((x * np.pi / self.image_size[0],
                                  y * np.pi / self.image_size[1]))
            else:
                for x in range(1, gx // 2 + 1):
                    freqs.append((x * np.pi / self.image_size[0],
                                  (y - gy) * np.pi / self.image_size[1]))
        self.frequencies = np.asarray(freqs, dtype=np.float64).reshape(-1, 2)

        n_freq = len(self.frequencies)
        coefficients = np.asarray(coefficients, dtype=np.float64).ravel()
        if coefficients.size < 4 * n_freq:
            raise ValueError(f"Fourier deformation needs {4 * n_freq} coefficients, "
                             f"got {coefficients.size}")
        # RawImage<dComplex>(n_freq, 2): index of dimension d, frequency i is
        # 2*(i + d*n_freq) + {real, imag}
        self.coefficients = coefficients[:4 * n_freq].reshape(2, n_freq, 2)

    def apply(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        t = xy @ self.frequencies.T                       # (..., n_freq)
        ct, st = np.cos(t), np.sin(t)
        shift = np.stack([ct @ self.coefficients[d, :, 0]
                          + st @ self.coefficients[d, :, 1] for d in range(2)],
                         axis=-1)
        return xy + shift


_DEFORMATION_MODELS = {
    "linear": Linear2DDeformation,
    "spline": Spline2DDeformation,
    "Fourier": Fourier2DDeformation,
}


def build_deformation(deformation_type, image_size, grid_size, coefficients):
    """
    Instantiate one of RELION's 2D deformation models.

        :param deformation_type (string) --> "linear", "spline" or "Fourier"
        :param image_size --> (W, H) of the tilt image, in pixels
        :param grid_size --> (grid_x, grid_y) from rlnTomoDeformationGridSizeX/Y
        :param coefficients --> the flat rlnTomoDeformationCoefficients vector
        :returns: an object with an ``apply(xy)`` method
    """
    if deformation_type not in _DEFORMATION_MODELS:
        raise ValueError(f"unknown deformation type {deformation_type!r}; RELION "
                         f"supports {sorted(_DEFORMATION_MODELS)}")
    cls = _DEFORMATION_MODELS[deformation_type]
    if cls is Linear2DDeformation:
        return cls(image_size, coefficients)
    return cls(image_size, grid_size, coefficients)


# --------------------------------------------------------------------------- #
#  Tilt-series geometry
# --------------------------------------------------------------------------- #

def projection_matrix_from_angles(xtilt, ytilt, zrot, xshift_angst, yshift_angst,
                                  pixel_size, tomo_size, tilt_image_size):
    """
    Reimplementation of RELION's ``Tomogram::setProjectionMatrix``::

        projectionMatrices[f] = s1 * s2 * r2 * r1 * r0 * s0

    The result maps a point of the tomogram, in *unbinned tilt-series pixels*,
    onto tilt image ``f``.  Components ``x, y`` are the pixel coordinates in the
    tilt image; component ``z`` is the depth along the beam, which RELION uses for
    the per-particle defocus offset.

        :param xtilt, ytilt, zrot --> per-frame angles in degrees, shape (F,)
        :param xshift_angst, yshift_angst --> per-frame specimen shifts, in Angstrom
        :param pixel_size --> unbinned tilt-series pixel size (A/px)
        :param tomo_size --> (SizeX, SizeY, SizeZ) of the tomogram, in pixels
        :param tilt_image_size --> (W, H) of a tilt image, in pixels
        :returns: (F, 4, 4) projection matrices
    """
    xtilt = np.atleast_1d(np.asarray(xtilt, dtype=np.float64))
    ytilt = np.atleast_1d(np.asarray(ytilt, dtype=np.float64))
    zrot = np.atleast_1d(np.asarray(zrot, dtype=np.float64))
    xshift_angst = np.atleast_1d(np.asarray(xshift_angst, dtype=np.float64))
    yshift_angst = np.atleast_1d(np.asarray(yshift_angst, dtype=np.float64))
    F = xtilt.shape[0]

    r0 = _axis_rotation((1.0, 0.0, 0.0), xtilt)
    r1 = _axis_rotation((0.0, 1.0, 0.0), ytilt)
    r2 = _axis_rotation((0.0, 0.0, 1.0), zrot)
    R = r2 @ r1 @ r0                                              # (F, 3, 3)

    # RELION uses C integer division for both centres, i.e. floor for positive sizes
    specimen_centre = np.array([int(tomo_size[0]) // 2,
                                int(tomo_size[1]) // 2,
                                int(tomo_size[2]) // 2], dtype=np.float64)
    image_centre = np.array([int(tilt_image_size[0]) // 2,
                             int(tilt_image_size[1]) // 2,
                             0.0], dtype=np.float64)

    specimen_shift = np.stack(
        [xshift_angst / pixel_size, yshift_angst / pixel_size, np.zeros(F)], axis=-1)

    P = np.zeros((F, 4, 4), dtype=np.float64)
    P[:, :3, :3] = R
    # s1 * s2 * (R @ s0):  translate by -centre first, then add both image offsets
    P[:, :3, 3] = -(R @ specimen_centre) + image_centre + specimen_shift
    P[:, 3, 3] = 1.0
    return P


@dataclass
class TiltSeriesGeometry:
    """
    Everything about one tomogram that is needed to place a particle on its tilt
    images.  Distances are in *unbinned tilt-series pixels*, angles in degrees.
    """

    name: str
    projection: np.ndarray          # (F, 4, 4) tomogram px -> tilt-image px
    tomo_size: np.ndarray           # (3,) tomogram dimensions in px
    tilt_image_size: np.ndarray     # (2,) tilt-image dimensions in px
    pixel_size: float               # unbinned tilt-series pixel size (A/px)
    handedness: float = -1.0
    defocus_slope: float = 1.0
    defocus_u: np.ndarray = None    # (F,)
    defocus_v: np.ndarray = None    # (F,)
    defocus_angle: np.ndarray = None
    ctf_scalefactor: np.ndarray = None
    phase_shift: np.ndarray = None
    pre_exposure: np.ndarray = None  # (F,) cumulative dose, e-/A^2
    nominal_tilt: np.ndarray = None  # (F,) rlnTomoNominalStageTiltAngle
    micrograph_name: np.ndarray = None   # (F,) path of each tilt image
    voltage: float = None
    spherical_aberration: float = None
    amplitude_contrast: float = None
    deformations: list = None            # (F,) 2D deformation models, or None

    @property
    def frame_count(self):
        return self.projection.shape[0]

    @property
    def has_deformations(self):
        return bool(self.deformations) and any(d is not None for d in self.deformations)

    @property
    def is_rotation_only(self):
        """
        True when the projection matrices carry no translation at all.

        RELION always writes the full affine (``-R*centre + image_centre + shift``),
        but WarpTools' ``ts_export_particles`` writes a literal zero translation
        column -- ``$"[{M.M11},{M.M12},{M.M13},0]"`` and ``"[0,0,0,1]"``. Such a
        matrix still gives the correct *orientation*, and the correct defocus offset
        (which is a difference, so the missing translation cancels), but projecting an
        absolute tomogram coordinate through it does not land anywhere meaningful.
        """
        return not np.any(np.abs(self.projection[:, :3, 3]) > 0)

    @property
    def centre(self):
        """Tomogram centre, the way RELION computes it (integer division)."""
        return np.array([int(self.tomo_size[0]) // 2,
                         int(self.tomo_size[1]) // 2,
                         int(self.tomo_size[2]) // 2], dtype=np.float64)

    def project(self, points):
        """
        Project 3D tomogram points onto every tilt image, exactly as
        ``Tomogram::projectPoint`` does: the linear projection matrix followed by
        the per-frame 2D deformation, when one was fitted.

            :param points --> (N, 3) for a point that does not move between frames,
                   or (N, F, 3) for a per-frame trajectory. Unbinned tilt-series px.
            :returns: (N, F, 3) with (x, y) the tilt-image pixel and z the depth
        """
        points = np.asarray(points, dtype=np.float64)
        R, t = self.projection[:, :3, :3], self.projection[None, :, :3, 3]

        if points.ndim == 2:
            out = np.einsum('fij,nj->nfi', R, points) + t
        elif points.ndim == 3:
            out = np.einsum('fij,nfj->nfi', R, points) + t
        else:
            raise ValueError(f"points must be (N, 3) or (N, F, 3), got {points.shape}")

        # A deformation warps the projected position only -- it never enters the
        # pose, so the depth (and therefore the defocus) is untouched
        if self.has_deformations:
            for f, deformation in enumerate(self.deformations):
                if deformation is not None:
                    out[:, f, :2] = deformation.apply(out[:, f, :2])
        return out

    def defocus_offset(self, projected):
        """
        ``Tomogram::getCtf``: the defocus change, in Angstrom, caused by the
        particle sitting off the tilt-axis plane::

            dz = handedness * pixelSize * defocusSlope * (z_particle - z_centre)

            :param projected --> (N, F, 3), output of :meth:`project`
            :returns: (N, F) defocus offset in Angstrom
        """
        z_centre = (self.projection[:, :3, :3] @ self.centre)[:, 2] + self.projection[:, 2, 3]
        return (self.handedness * self.pixel_size * self.defocus_slope
                * (projected[..., 2] - z_centre[None, :]))

    def visibility(self, projected, radius):
        """
        ``Tomogram::isVisible``: the whole box must fall inside the tilt image.

            :param projected --> (N, F, 3)
            :param radius --> half the box size, in unbinned tilt-series pixels
            :returns: (N, F) boolean
        """
        x, y = projected[..., 0], projected[..., 1]
        return ((x > radius) & (x < self.tilt_image_size[0] - radius)
                & (y > radius) & (y < self.tilt_image_size[1] - radius))


# --------------------------------------------------------------------------- #
#  STAR reading helpers
# --------------------------------------------------------------------------- #

def _read_star(path):
    """
    starfile.read that always yields a dict of DataFrames.  Loop-less blocks --
    ``data_optimisation_set`` is one -- come back as a plain ``dict`` (starfile
    >= 0.5) or a ``Series`` (older releases), so both are promoted to a single-row
    frame rather than silently dropped.  Dropping them would lose the optimisation
    set entirely, which is the one block that names every other file.
    """
    out = starfile.read(path, always_dict=True)
    blocks = {}
    for key, value in out.items():
        if isinstance(value, pd.DataFrame):
            blocks[key] = value
        elif isinstance(value, pd.Series):
            blocks[key] = value.to_frame().T
        elif isinstance(value, dict):
            blocks[key] = pd.DataFrame([value])
    return blocks


# The labels that identify an optimisation set. RELION names the block
# ``optimisation_set``, but a hand-written or re-exported file may leave it
# unnamed, so the labels -- not the block name -- are what we key on.
_OPTIMISATION_SET_LABEL = "rlnTomoParticlesFile"


def _optimisation_block(blocks):
    """The optimisation-set block of a parsed STAR file, or None."""
    for df in blocks.values():
        if _OPTIMISATION_SET_LABEL in df.columns:
            return df
    return None


def _col(df, name, default=None, dtype=np.float64):
    """Fetch a column as a numpy array, or broadcast a default when it is absent."""
    if name in df.columns:
        return np.asarray(df[name].to_numpy(), dtype=dtype)
    if default is None:
        return None
    return np.full(len(df), default, dtype=dtype)


_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")


def _parse_vector(value):
    """
    Parse one STAR cell holding a vector, e.g. ``[1.000000,0.000000,0.000000]``.
    Values the reader already turned into a list/array are passed straight through.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float64).ravel()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.zeros(0, dtype=np.float64)
    text = str(value).replace("d", "e").replace("D", "E")
    return np.asarray([float(v) for v in _NUMBER_RE.findall(text)], dtype=np.float64)


def _parse_vector_column(df, name, length):
    """
    Parse a STAR column whose entries are bracketed vectors, e.g.
    ``rlnTomoProjX`` -> ``[1.000000,0.000000,0.000000,0.000000]`` or
    ``rlnTomoVisibleFrames`` -> ``[1,1,0,1]``.
    """
    if name not in df.columns:
        return None

    out = np.zeros((len(df), length), dtype=np.float64)
    for i, value in enumerate(df[name].to_numpy()):
        vals = _parse_vector(value)
        if vals.size != length:
            raise ValueError(
                f"{name} row {i} has {vals.size} entries, expected {length}")
        out[i] = vals
    return out


# RELION's deformation label strings differ between versions; accept either
_DEFORMATION_GRID_X = ("rlnTomoDeformationGridSizeX", "rlnDeformationGridSizeX")
_DEFORMATION_GRID_Y = ("rlnTomoDeformationGridSizeY", "rlnDeformationGridSizeY")
_DEFORMATION_TYPE = ("rlnTomoDeformationType", "rlnDeformationType")
_DEFORMATION_COEFFS = ("rlnTomoDeformationCoefficients", "rlnDeformationCoefficients")


def _pick(container, names):
    """First of ``names`` present in a DataFrame's columns or a Series' index."""
    keys = container.columns if isinstance(container, pd.DataFrame) else container.index
    for name in names:
        if name in keys:
            return name
    return None


def _read_deformations(global_row, ts, image_size, n_frames, tomo_name):
    """
    Build the per-frame 2D deformation models of one tomogram, if
    ``relion_tomo_align --deformation`` was run.

        :returns: list of n_frames models (entries may be None), or None
    """
    gx_label = _pick(global_row, _DEFORMATION_GRID_X)
    gy_label = _pick(global_row, _DEFORMATION_GRID_Y)
    coeff_label = _pick(ts, _DEFORMATION_COEFFS)
    if gx_label is None or gy_label is None or coeff_label is None:
        return None

    type_label = _pick(global_row, _DEFORMATION_TYPE)
    deformation_type = str(global_row[type_label]) if type_label else "spline"
    grid = (int(global_row[gx_label]), int(global_row[gy_label]))

    if deformation_type == "Fourier":
        warnings.warn(
            f"tomogram '{tomo_name}' uses the Fourier deformation model, whose "
            f"coefficient layout is ambiguous in RELION's own source; the crop "
            f"centres it produces have not been checked against real data",
            RuntimeWarning)

    values = ts[coeff_label].to_numpy()
    out = []
    for f in range(n_frames):
        coeffs = _parse_vector(values[f])
        out.append(build_deformation(deformation_type, image_size, grid, coeffs)
                   if coeffs.size else None)
    return out


def _resolve(path, relative_to):
    """Resolve a STAR-file path, which RELION writes relative to the project root."""
    path = str(path)
    if os.path.isabs(path) or os.path.exists(path):
        return path
    for base in (relative_to, os.path.dirname(relative_to),
                 os.path.dirname(os.path.dirname(relative_to))):
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return candidate
    return path


def read_optimisation_set(path):
    """
    Read a RELION-5 ``optimisation_set.star`` and resolve the files it points to.

        :param path (string) --> path to the optimisation set
        :returns: dict with keys 'particles', 'tomograms', 'trajectories',
                  'manifolds', 'reference_map' (missing entries are None)
    """
    blocks = _read_star(path)
    df = _optimisation_block(blocks)
    if df is None:
        raise ValueError(
            f"{path} has no block with {_OPTIMISATION_SET_LABEL}; is it a RELION "
            f"optimisation set?")
    row = df.iloc[0]
    base = os.path.dirname(os.path.abspath(path))

    keys = {
        "particles": "rlnTomoParticlesFile",
        "tomograms": "rlnTomoTomogramsFile",
        "trajectories": "rlnTomoTrajectoriesFile",
        "manifolds": "rlnTomoManifoldsFile",
        "reference_map": "rlnTomoReferenceMap",
    }
    out = {}
    for key, label in keys.items():
        value = row.get(label, None)
        out[key] = _resolve(value, base) if isinstance(value, str) and value else None
    return out


def read_trajectories_star(path):
    """
    Read a RELION ``motion.star`` (``rlnTomoTrajectoriesFile``), as written by
    ``relion_tomo_align --motion``.

    The file holds one data block per particle, named after its
    ``rlnTomoParticleName``, with one row per tilt frame carrying
    ``rlnOriginX/Y/ZAngst``.  ``Trajectory::getShiftsInPix`` **adds** them to the
    particle position: ``out[f] = origin + shifts_Ang[f] / pixelSize``.

        :param path (string) --> path to motion.star
        :returns: dict mapping rlnTomoParticleName -> (F, 3) offsets in Angstrom
    """
    blocks = _read_star(path)
    cols = ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst")

    out = {}
    for name, df in blocks.items():
        if not all(c in df.columns for c in cols):
            continue
        shifts = np.stack([df[c].to_numpy(dtype=np.float64) for c in cols], axis=-1)
        # RELION names the block after the particle, but tolerate a file that
        # instead carries the name as a column
        if "rlnTomoParticleName" in df.columns:
            key = str(df["rlnTomoParticleName"].to_numpy()[0])
        else:
            key = str(name)
        out[key] = shifts

    if not out:
        raise ValueError(
            f"{path} has no per-particle block with rlnOriginX/Y/ZAngst; is it a "
            f"RELION trajectories file?")
    return out


def _tilt_image_size_from_header(micrograph_name, star_dir):
    """Best-effort read of the tilt-image dimensions from the image header."""
    if micrograph_name is None:
        return None
    try:
        from xmipp_metadata.image_handler.image_handler import ImageHandler

        name = str(micrograph_name)
        if "@" in name:
            name = name.split("@", 1)[1]
        name = _resolve(name, star_dir)
        if not os.path.exists(name):
            return None
        dims = ImageHandler(name).getDimensions()
    except Exception:
        return None

    if dims is None:
        return None
    dims = np.asarray(dims).ravel()
    if dims.size < 3:
        return None
    # ImageHandler reports (nz, ny, nx); RELION's tilt-image centre is (W/2, H/2)
    return np.array([int(dims[2]), int(dims[1])], dtype=np.int64)


def read_tomograms_star(path, tilt_image_size=None):
    """
    Read a RELION-4/5 tomograms STAR file into :class:`TiltSeriesGeometry` objects.

    Handles both layouts RELION supports:
      * RELION-5: a ``global`` block whose ``rlnTomoTiltSeriesStarFile`` column
        points at one STAR file per tomogram.
      * RELION-4: a ``global`` block followed by one block per tomogram, named
        after ``rlnTomoName``, in the same file.

    And both parameterisations of the geometry:
      * ``rlnTomoProjX/Y/Z/W``, the explicit 4x4 matrix (older files), used as-is.
      * ``rlnTomoXTilt/YTilt/ZRot/XShiftAngst/YShiftAngst``, from which the matrix
        is rebuilt exactly as ``Tomogram::setProjectionMatrix`` does.

        :param path (string) --> path to tomograms.star
        :param tilt_image_size (tuple - Optional) --> (W, H) override, in pixels.
               When omitted it is taken from the tilt-image header, falling back
               to the tomogram's X/Y size.
        :returns: dict mapping rlnTomoName -> TiltSeriesGeometry
    """
    path = os.path.abspath(path)
    star_dir = os.path.dirname(path)
    blocks = _read_star(path)

    if "global" not in blocks:
        raise ValueError(f"{path} has no 'global' block; is it a tomograms STAR file?")
    g = blocks["global"]

    if "rlnTomoName" not in g.columns:
        raise ValueError(f"{path} has no rlnTomoName column")

    # Inline layout (RELION-4, and what WarpTools writes): the per-tomogram tables
    # follow the global block in the same file. RELION matches them by position
    # (``allTables[t+1]``), so fall back to that when the block name does not match
    # rlnTomoName -- Warp, for one, does not always keep the two in step.
    inline_blocks = [k for k in blocks if k != "global"]

    out = {}
    for i in range(len(g)):
        row = g.iloc[i]
        name = str(row["rlnTomoName"])

        # ---- locate the per-tomogram tilt-series table -------------------- #
        if "rlnTomoTiltSeriesStarFile" in g.columns:
            ts_path = _resolve(row["rlnTomoTiltSeriesStarFile"], star_dir)
            ts_blocks = _read_star(ts_path)
            ts = ts_blocks.get(name, next(iter(ts_blocks.values())))
            ts_dir = os.path.dirname(os.path.abspath(ts_path))
        elif name in blocks:
            ts = blocks[name]
            ts_dir = star_dir
        elif i < len(inline_blocks):
            ts = blocks[inline_blocks[i]]
            ts_dir = star_dir
        else:
            raise ValueError(
                f"tomogram '{name}' has neither rlnTomoTiltSeriesStarFile nor a "
                f"data_{name} block in {path}, and there is no {i + 1}-th data "
                f"block to fall back on")

        F = len(ts)
        pixel_size = float(row.get("rlnTomoTiltSeriesPixelSize", np.nan))
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError(
                f"tomogram '{name}' has no usable rlnTomoTiltSeriesPixelSize")

        tomo_size = np.array([row.get("rlnTomoSizeX", np.nan),
                              row.get("rlnTomoSizeY", np.nan),
                              row.get("rlnTomoSizeZ", np.nan)], dtype=np.float64)

        micrograph_name = (ts["rlnMicrographName"].to_numpy()
                           if "rlnMicrographName" in ts.columns else None)

        # ---- tilt image dimensions ---------------------------------------- #
        if tilt_image_size is not None:
            image_size = np.asarray(tilt_image_size, dtype=np.int64)
        else:
            image_size = _tilt_image_size_from_header(
                micrograph_name[0] if micrograph_name is not None else None, ts_dir)
            if image_size is None:
                if not np.all(np.isfinite(tomo_size[:2])):
                    raise ValueError(
                        f"cannot determine the tilt-image size of '{name}': no "
                        f"readable tilt image and no rlnTomoSizeX/Y. Pass "
                        f"tilt_image_size=(W, H) explicitly.")
                image_size = tomo_size[:2].astype(np.int64)

        # ---- projection matrices ------------------------------------------ #
        proj_rows = [_parse_vector_column(ts, lbl, 4)
                     for lbl in ("rlnTomoProjX", "rlnTomoProjY",
                                 "rlnTomoProjZ", "rlnTomoProjW")]
        if all(r is not None for r in proj_rows):
            projection = np.stack(proj_rows, axis=1)          # (F, 4, 4)
        else:
            if "rlnTomoYTilt" not in ts.columns:
                raise ValueError(
                    f"tomogram '{name}' has neither rlnTomoProjX/Y/Z/W nor "
                    f"rlnTomoYTilt: no tilt-series geometry to work with")
            if not np.all(np.isfinite(tomo_size)):
                raise ValueError(
                    f"tomogram '{name}' needs rlnTomoSizeX/Y/Z to rebuild the "
                    f"projection matrices from rlnTomoXTilt/YTilt/ZRot")
            projection = projection_matrix_from_angles(
                _col(ts, "rlnTomoXTilt", 0.0),
                _col(ts, "rlnTomoYTilt", 0.0),
                _col(ts, "rlnTomoZRot", 0.0),
                _col(ts, "rlnTomoXShiftAngst", 0.0),
                _col(ts, "rlnTomoYShiftAngst", 0.0),
                pixel_size, tomo_size, image_size)

        out[name] = TiltSeriesGeometry(
            name=name,
            projection=projection,
            tomo_size=tomo_size,
            tilt_image_size=image_size,
            pixel_size=pixel_size,
            handedness=float(row.get("rlnTomoHand", -1.0)),
            defocus_slope=float(row.get("rlnTomoDefocusSlope", 1.0)),
            defocus_u=_col(ts, "rlnDefocusU", 0.0),
            defocus_v=_col(ts, "rlnDefocusV", 0.0),
            defocus_angle=_col(ts, "rlnDefocusAngle", 0.0),
            ctf_scalefactor=_col(ts, "rlnCtfScalefactor", 1.0),
            phase_shift=_col(ts, "rlnPhaseShift", 0.0),
            pre_exposure=_col(ts, "rlnMicrographPreExposure", 0.0),
            nominal_tilt=_col(ts, "rlnTomoNominalStageTiltAngle", np.nan),
            micrograph_name=micrograph_name,
            voltage=float(row.get("rlnVoltage", np.nan)),
            spherical_aberration=float(row.get("rlnSphericalAberration", np.nan)),
            amplitude_contrast=float(row.get("rlnAmplitudeContrast", np.nan)),
            deformations=_read_deformations(row, ts, image_size, F, name),
        )
        if F != out[name].frame_count:
            raise ValueError(
                f"tomogram '{name}': {F} rows but {out[name].frame_count} matrices")

    return out


# --------------------------------------------------------------------------- #
#  Particle side
# --------------------------------------------------------------------------- #

def _particle_positions(df, geom):
    """
    ``ParticleSet::getPosition``: the particle centre in unbinned tilt-series
    pixels, with the refined subtomogram origin shift applied::

        pos = coord - (A_subtomogram @ origin_angst) / tiltSeriesPixelSize

    Note the shift is *subtracted*, and is rotated by the pseudo-subtomogram
    matrix first, because RELION stores it in the subtomogram frame.
    """
    n = len(df)
    apix = geom.pixel_size

    centred = all(c in df.columns for c in ("rlnCenteredCoordinateXAngst",
                                            "rlnCenteredCoordinateYAngst",
                                            "rlnCenteredCoordinateZAngst"))
    if centred:
        pos = np.stack([_col(df, "rlnCenteredCoordinateXAngst"),
                        _col(df, "rlnCenteredCoordinateYAngst"),
                        _col(df, "rlnCenteredCoordinateZAngst")], axis=-1) / apix
        pos = pos + geom.centre[None, :]
    elif all(c in df.columns for c in ("rlnCoordinateX", "rlnCoordinateY",
                                       "rlnCoordinateZ")):
        pos = np.stack([_col(df, "rlnCoordinateX"),
                        _col(df, "rlnCoordinateY"),
                        _col(df, "rlnCoordinateZ")], axis=-1)
    else:
        raise ValueError(
            "particles table has neither rlnCenteredCoordinateX/Y/ZAngst nor "
            "rlnCoordinateX/Y/Z")

    offset = np.stack([_col(df, "rlnOriginXAngst", 0.0),
                       _col(df, "rlnOriginYAngst", 0.0),
                       _col(df, "rlnOriginZAngst", 0.0)], axis=-1)

    if "rlnTomoSubtomogramRot" in df.columns:
        A_sub = relion_angles_to_matrix(_col(df, "rlnTomoSubtomogramRot", 0.0),
                                        _col(df, "rlnTomoSubtomogramTilt", 0.0),
                                        _col(df, "rlnTomoSubtomogramPsi", 0.0))
    else:
        A_sub = np.broadcast_to(np.eye(3), (n, 3, 3))

    # ``delta`` is the displacement the origin shift applies to the coordinate, kept
    # separately because it is what the "from_origin" shift mode projects: being a
    # *difference* of positions, it survives a projection matrix with no translation.
    delta = -np.einsum('nij,nj->ni', A_sub, offset) / apix
    return pos + delta, A_sub, delta


def is_relion_tomo_star(blocks):
    """
    True when this STAR file is a RELION tomography particles file that still
    needs the tilt-series geometry folded in.

        :param blocks --> dict of DataFrames (or a single DataFrame)
        :returns: bool
    """
    if isinstance(blocks, pd.DataFrame):
        frames = [blocks]
    else:
        frames = list(blocks.values())
    for df in frames:
        if not isinstance(df, pd.DataFrame):
            continue
        cols = set(df.columns)
        # an optimisation set is not a particles table itself, but it names one
        if _OPTIMISATION_SET_LABEL in cols:
            return True
        if "rlnTomoName" in cols and (
                {"rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateZAngst"} & cols
                or "rlnCoordinateZ" in cols):
            return True
    return False


def tomo_star_to_tilt_particles(
        particles_star,
        tomograms_star=None,
        *,
        box_size=None,
        binning=None,
        shifts="auto",
        shift_units="angstrom",
        tilt_image_size=None,
        visibility="auto",
        trajectories=None,
        chunk_size=20000,
        keep_columns=("rlnRandomSubset", "rlnClassNumber", "rlnGroupNumber",
                      "rlnOpticsGroup", "rlnTomoParticleName", "rlnTomoParticleId"),
):
    """
    Expand a RELION tomography particles STAR file into one row per
    (particle, visible tilt image), carrying the *composed* 2D alignment.

    The composition, straight out of ``reconstruct_particle.cpp``::

        projPart_f = projCut_f @ particleToTomo
                   = [P_f with its shift set so the particle lands on the box centre]
                     @ [Ts(pos) @ R(A_sub @ A_part) @ Tc(-s/2)]

    Expanding the affine parts, every translation cancels and what is left is::

        u  ->  (R_f @ A_sub @ A_part) @ (u - s/2) + s/2

    i.e. a pure rotation about the box centre.  Therefore

      * the per-tilt Euler triplet is ``matrix_to_relion_angles(R_f @ A_sub @ A_part)``,
        where ``R_f`` is the 3x3 block of the tilt-series projection matrix;
      * the only translation left is the difference between the integer pixel the
        box is cropped at and the exact projected particle centre.  RELION removes
        that residual with a Fourier phase shift during extraction, so its own 2D
        stacks are already perfectly centred.

    ``shifts`` controls which of those two situations is assumed:

      * ``"zero"``        -- the images are already centred (RELION 2D stacks).
      * ``"residual"``    -- the images will be cropped at the integer pixel given
                             by ``rlnCoordinateX/Y``, so the sub-pixel remainder is
                             written to ``rlnOriginX/YAngst``.
      * ``"from_origin"`` -- the images were extracted centred on ``rlnCoordinateX/Y/Z``
                             and ``rlnOriginX/Y/ZAngst`` is a 3D correction refined
                             *since* that extraction. Its per-tilt 2D effect is written
                             out. This is what a refinement run on top of already-extracted
                             2D stacks produces, and dropping it would discard the whole
                             translational part of that refinement. It stays exact on
                             rotation-only projection matrices, because projecting a
                             *difference* of positions cancels the missing translation.
      * ``"auto"``        -- ``"zero"`` when the input carries ``rlnTomoVisibleFrames``
                             (2D stacks were written), ``"residual"`` otherwise. Note
                             ``auto`` never selects ``from_origin``: whether an origin is
                             a correction *since* extraction or was already folded into
                             the extraction cannot be told from the file, so it has to be
                             asked for. A warning is raised when the choice looks wrong.

    Sign convention for the residual follows RELION everywhere: the true particle
    centre is ``coordinate - origin``, exactly as in ``ParticleSet::getPosition``.

    ``rlnCoordinateX/Y`` and the residual are expressed on the *unbinned* tilt-series
    pixel grid, because that is the grid RELION crops on (``integralShift[f] =
    round(centers[f]) - s/2`` happens before the Fourier downsampling).  Since
    ``rlnOriginX/YAngst`` is in Angstrom it is independent of any later binning.

        :param particles_star (string) --> particles STAR, or an optimisation_set.star
        :param tomograms_star (string - Optional) --> tomograms STAR; taken from the
               optimisation set, or looked for next to the particles file, if omitted
        :param box_size (int - Optional) --> box edge in *unbinned tilt-series* pixels,
               used only to decide visibility when the input does not store it.
               Defaults to rlnImageSize * binning from the optics block.
        :param binning (float - Optional) --> output pixel size / unbinned tilt-series
               pixel size. Defaults to rlnImagePixelSize / rlnTomoTiltSeriesPixelSize,
               falling back to 1.0
        :param shifts (string) --> "auto", "zero" or "residual"
        :param shift_units (string) --> "angstrom" writes rlnOriginX/YAngst, RELION's
               own convention; "pixel" writes rlnOriginX/Y in *output* pixels instead,
               for consumers that expect pixels
        :param tilt_image_size (tuple - Optional) --> (W, H) override in pixels
        :param visibility (string) --> "auto", "stored" or "computed"
        :param trajectories (string or dict - Optional) --> a RELION motion.star, or
               an already-loaded dict of per-particle (F, 3) offsets in Angstrom in
               the tomogram frame, keyed by rlnTomoParticleName. Loaded automatically
               from the optimisation set when one is given as input.
        :param chunk_size (int) --> particles processed per batch, bounds peak memory
        :param keep_columns (tuple) --> particle columns copied through unchanged
        :returns: pandas DataFrame with RELION labels, one row per tilt image, plus a
                  dense 1-based ``subtomo_labels`` column grouping rows by particle
    """
    if shifts not in ("auto", "zero", "residual", "from_origin"):
        raise ValueError(
            f"shifts must be auto/zero/residual/from_origin, got {shifts!r}")
    if shift_units not in ("angstrom", "pixel"):
        raise ValueError(f"shift_units must be angstrom/pixel, got {shift_units!r}")
    if visibility not in ("auto", "stored", "computed"):
        raise ValueError(f"visibility must be auto/stored/computed, got {visibility!r}")

    particles_star = os.path.abspath(particles_star)
    blocks = _read_star(particles_star)

    # An optimisation set points at the real files
    if _optimisation_block(blocks) is not None:
        opt = read_optimisation_set(particles_star)
        if opt["particles"] is None:
            raise ValueError(f"{particles_star} has no rlnTomoParticlesFile")
        if tomograms_star is None:
            tomograms_star = opt["tomograms"]
        if trajectories is None and opt["trajectories"] is not None:
            trajectories = opt["trajectories"]
        particles_star = opt["particles"]
        blocks = _read_star(particles_star)

    if isinstance(trajectories, (str, os.PathLike)):
        trajectories = read_trajectories_star(trajectories)

    star_dir = os.path.dirname(particles_star)
    optics = blocks.get("optics")
    if "particles" in blocks:
        parts = blocks["particles"]
    else:
        parts = max(blocks.values(), key=lambda d: len(d.columns))

    if "rlnTomoName" not in parts.columns:
        raise ValueError(f"{particles_star} has no rlnTomoName column")

    # ---- find the tomograms STAR ----------------------------------------- #
    if tomograms_star is None:
        for candidate in ("tomograms.star", "optimisation_set.star"):
            guess = os.path.join(star_dir, candidate)
            if os.path.exists(guess):
                if candidate == "optimisation_set.star":
                    tomograms_star = read_optimisation_set(guess)["tomograms"]
                else:
                    tomograms_star = guess
                break
    if tomograms_star is None:
        # WarpTools names its export <prefix>_tomograms.star, next to <prefix>.star
        stem = os.path.splitext(os.path.basename(particles_star))[0]
        candidates = [os.path.join(star_dir, f"{stem}_tomograms.star")]
        candidates += sorted(glob(os.path.join(star_dir, "*_tomograms.star")))
        for guess in candidates:
            if os.path.exists(guess):
                tomograms_star = guess
                break
    if tomograms_star is None:
        raise ValueError(
            f"no tomograms STAR file given, and none found next to {particles_star}. "
            f"The tilt-series geometry cannot be recovered without it -- pass "
            f"tomograms_star=... explicitly.")

    geoms = read_tomograms_star(tomograms_star, tilt_image_size=tilt_image_size)

    # ---- optics ----------------------------------------------------------- #
    if optics is not None and "rlnOpticsGroup" in parts.columns \
            and "rlnOpticsGroup" in optics.columns:
        optics_cols = [c for c in optics.columns
                       if c not in parts.columns or c == "rlnOpticsGroup"]
        parts = parts.merge(optics[optics_cols], on="rlnOpticsGroup", how="left")

    # ---- output sampling --------------------------------------------------- #
    # RELION's extraction takes a single --bin for the whole data set, so binning is
    # a scalar here too; guard against tomograms that disagree on the tilt-series
    # sampling, which would make that scalar meaningless.
    tilt_apix = np.array([g.pixel_size for g in geoms.values()])
    apix_ts_ref = float(tilt_apix[0])
    if not np.allclose(tilt_apix, apix_ts_ref, rtol=1e-6):
        warnings.warn(
            "the tomograms do not share a common rlnTomoTiltSeriesPixelSize; "
            f"using {apix_ts_ref} to derive the binning factor", RuntimeWarning)

    if binning is None:
        if "rlnImagePixelSize" in parts.columns:
            binning = float(np.median(parts["rlnImagePixelSize"].to_numpy())) / apix_ts_ref
        else:
            binning = 1.0
    binning = float(binning)

    if box_size is None and "rlnImageSize" in parts.columns:
        box_size = float(np.median(parts["rlnImageSize"].to_numpy())) * binning

    has_stack2d = "rlnTomoVisibleFrames" in parts.columns
    requested_shifts = shifts
    if shifts == "auto":
        shifts = "zero" if has_stack2d else "residual"

    # Dropping a refined origin is silent and expensive -- the map just comes out worse --
    # so say so loudly when the input looks like a refinement on top of extracted stacks.
    if shifts == "zero":
        origin_cols = [c for c in ("rlnOriginXAngst", "rlnOriginYAngst",
                                   "rlnOriginZAngst") if c in parts.columns]
        if origin_cols:
            largest = float(np.abs(parts[origin_cols].to_numpy(dtype=np.float64)).max())
            if largest > 1e-6:
                how = ("auto-selected" if requested_shifts == "auto" else "requested")
                warnings.warn(
                    f"shifts='zero' ({how}) but the particles carry non-zero origin "
                    f"shifts (up to {largest:.3g} A). If those were refined *after* the "
                    f"2D stacks were extracted -- which is what a refinement run on "
                    f"extracted stacks produces -- they are being discarded, and the "
                    f"whole translational part of that refinement with them. Pass "
                    f"shifts='from_origin' to project them onto each tilt instead.",
                    RuntimeWarning)
    if visibility == "auto":
        visibility = "stored" if has_stack2d else "computed"
    if visibility == "stored" and not has_stack2d:
        raise ValueError("visibility='stored' but the input has no "
                         "rlnTomoVisibleFrames column")
    if visibility == "computed" and has_stack2d:
        warnings.warn(
            "recomputing visibility for a data set that already has extracted 2D "
            "stacks: any frame this disagrees with RELION on will misalign the "
            "slice indices written to rlnImageName", RuntimeWarning)

    # WarpTools writes rotation-only projection matrices. The orientation and the
    # defocus gradient survive that, but anything derived from an *absolute* projected
    # position does not -- so refuse the two modes that would silently produce
    # nonsense rather than let them through.
    rotation_only = [name for name, geom in geoms.items() if geom.is_rotation_only]
    if rotation_only:
        detail = (f"{len(rotation_only)} tomogram(s), e.g. '{rotation_only[0]}', have "
                  f"projection matrices with a zero translation column (WarpTools "
                  f"writes them this way)")
        if shifts == "residual":
            raise ValueError(
                f"{detail}, so the projected particle centre is not a real tilt-image "
                f"position and shifts='residual' would be meaningless. Use "
                f"shifts='zero' -- the images such a pipeline produces are already "
                f"centred.")
        if visibility == "computed":
            raise ValueError(
                f"{detail}, so visibility cannot be recomputed from the projected "
                f"position. Use visibility='stored' (rlnTomoVisibleFrames).")
        warnings.warn(
            f"{detail}. Orientations and per-tilt defocus are still correct, but "
            f"rlnCoordinateX/Y in the output are not real tilt-image coordinates and "
            f"must not be used to crop from the tilt series.", RuntimeWarning)

    # Global particle index -> dense 1-based subtomogram label
    parts = parts.reset_index(drop=True)
    parts["__subtomo_label__"] = np.arange(1, len(parts) + 1, dtype=np.int64)

    frames = []
    for tomo_name, group in parts.groupby("rlnTomoName", sort=False):
        if str(tomo_name) not in geoms:
            raise ValueError(
                f"tomogram '{tomo_name}' is referenced by the particles file but "
                f"missing from {tomograms_star}")
        geom = geoms[str(tomo_name)]
        for start in range(0, len(group), chunk_size):
            frames.append(_expand_tomogram(
                group.iloc[start:start + chunk_size], geom,
                box_size=box_size, binning=binning, shifts=shifts,
                shift_units=shift_units, visibility=visibility,
                trajectories=trajectories, keep_columns=keep_columns,
                has_stack2d=has_stack2d))

    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    return out


def _expand_tomogram(df, geom, *, box_size, binning, shifts, shift_units, visibility,
                     trajectories, keep_columns, has_stack2d):
    """Expand the particles of a single tomogram. See :func:`tomo_star_to_tilt_particles`."""
    n = len(df)
    F = geom.frame_count
    apix_ts = geom.pixel_size
    apix_out = apix_ts * binning

    pos, A_sub, delta = _particle_positions(df, geom)

    # ---- optional per-tilt particle motion -------------------------------- #
    traj = None
    if trajectories:
        if "rlnTomoParticleName" not in df.columns:
            raise ValueError(
                "trajectories were given but the particles table has no "
                "rlnTomoParticleName column to key them on")
        traj = np.zeros((n, F, 3), dtype=np.float64)
        missing = 0
        for i, name in enumerate(df["rlnTomoParticleName"].to_numpy()):
            shift = trajectories.get(str(name))
            if shift is None:
                missing += 1
                continue
            shift = np.asarray(shift, dtype=np.float64)
            if shift.shape != (F, 3):
                raise ValueError(
                    f"trajectory for particle '{name}' has shape {shift.shape}, "
                    f"expected ({F}, 3)")
            traj[i] = shift / apix_ts
        if missing:
            warnings.warn(
                f"{missing} of {n} particles of tomogram '{geom.name}' have no "
                f"trajectory; they are left unshifted", RuntimeWarning)

    # ---- project ----------------------------------------------------------- #
    projected = geom.project(pos if traj is None else pos[:, None, :] + traj)

    # ---- composed orientation --------------------------------------------- #
    A_part = relion_angles_to_matrix(_col(df, "rlnAngleRot", 0.0),
                                     _col(df, "rlnAngleTilt", 0.0),
                                     _col(df, "rlnAnglePsi", 0.0))
    A = A_sub @ A_part                                                  # (n, 3, 3)
    A_tot = np.einsum('fij,njk->nfik', geom.projection[:, :3, :3], A)   # (n, F, 3, 3)
    rot, tilt, psi = matrix_to_relion_angles(A_tot)                     # (n, F) each

    # ---- visibility -------------------------------------------------------- #
    if visibility == "stored":
        visible = _parse_vector_column(df, "rlnTomoVisibleFrames", F).astype(bool)
    else:
        if box_size is None:
            raise ValueError(
                "visibility must be computed but box_size was not given and could "
                "not be derived from rlnImageSize; pass box_size=<box edge in "
                "unbinned tilt-series pixels>")
        visible = geom.visibility(projected, 0.5 * float(box_size))

    # ---- shifts ------------------------------------------------------------ #
    # The crop happens on the unbinned tilt-series grid, exactly as RELION does it
    # (integralShift[f] = round(centers[f]) - s/2, before the Fourier downsampling)
    centre_px = projected[..., :2]
    coord_int = np.rint(centre_px)
    if shifts == "zero":
        origin_angst = np.zeros_like(centre_px)
    elif shifts == "from_origin":
        # The images were extracted centred on rlnCoordinateX/Y/Z; rlnOriginX/Y/ZAngst is
        # the 3D correction refined *since*. Its per-tilt 2D effect is the projection of
        # that displacement -- and because this is a difference of two positions, the
        # translation column of the projection matrix cancels, so it stays exact on the
        # rotation-only matrices WarpTools writes.
        delta_2d = np.einsum('fij,nj->nfi', geom.projection[:, :3, :3], delta)[..., :2]
        # RELION: true centre = coordinate - origin, and the refined centre sits at
        # (extraction centre + delta_2d), so origin = -delta_2d
        origin_angst = -delta_2d * apix_ts
        # The crop already happened, at the extraction centre, and nothing here is
        # instructing a new one -- so report that centre unrounded rather than rounding it
        # as the "residual" mode does. Rounding would have to be folded into the origin,
        # which would stop it being a pure difference and so stop it working on a
        # rotation-only geometry, which is the whole point of this mode.
        coord_int = centre_px - delta_2d
    else:
        # RELION: true centre = coordinate - origin  =>  origin = coordinate - centre
        origin_angst = (coord_int - centre_px) * apix_ts

    # ---- CTF --------------------------------------------------------------- #
    dz = geom.defocus_offset(projected)                                 # (n, F)
    defocus_u = geom.defocus_u[None, :] + dz
    defocus_v = geom.defocus_v[None, :] + dz

    # ---- flatten ------------------------------------------------------------ #
    rows = np.nonzero(visible)                       # (particle idx, frame idx)
    pi, fi = rows
    if pi.size == 0:
        return pd.DataFrame()

    # index of each tilt image inside its particle's 2D stack: RELION writes only
    # the visible slices, in frame order, so this is a running count per particle
    slice_index = (np.cumsum(visible, axis=1) - 1)[pi, fi]

    out = {
        "rlnTomoName": np.repeat(str(geom.name), pi.size),
        "subtomo_labels": df["__subtomo_label__"].to_numpy()[pi],
        "rlnAngleRot": rot[pi, fi],
        "rlnAngleTilt": tilt[pi, fi],
        "rlnAnglePsi": psi[pi, fi],
        "rlnCoordinateX": coord_int[pi, fi, 0],
        "rlnCoordinateY": coord_int[pi, fi, 1],
        "rlnDefocusU": defocus_u[pi, fi],
        "rlnDefocusV": defocus_v[pi, fi],
        "rlnDefocusAngle": geom.defocus_angle[fi],
        "rlnCtfScalefactor": geom.ctf_scalefactor[fi],
        "rlnPhaseShift": geom.phase_shift[fi],
        "rlnMicrographPreExposure": geom.pre_exposure[fi],
        "rlnImagePixelSize": np.full(pi.size, apix_out),
        "rlnTomoTiltSeriesPixelSize": np.full(pi.size, apix_ts),
        "rlnTomoFrameIndex": fi + 1,
    }

    if shift_units == "angstrom":
        out["rlnOriginXAngst"] = origin_angst[pi, fi, 0]
        out["rlnOriginYAngst"] = origin_angst[pi, fi, 1]
    else:
        out["rlnOriginX"] = origin_angst[pi, fi, 0] / apix_out
        out["rlnOriginY"] = origin_angst[pi, fi, 1] / apix_out

    # Microscope constants live in the tomograms file, but older data sets keep them
    # only in the particles' optics group, so fall back to whatever is there
    for label, value in (("rlnVoltage", geom.voltage),
                         ("rlnSphericalAberration", geom.spherical_aberration),
                         ("rlnAmplitudeContrast", geom.amplitude_contrast)):
        if value is not None and np.isfinite(value):
            out[label] = np.full(pi.size, value)
        elif label in df.columns:
            out[label] = df[label].to_numpy()[pi]
    if geom.nominal_tilt is not None and np.any(np.isfinite(geom.nominal_tilt)):
        out["rlnTomoNominalStageTiltAngle"] = geom.nominal_tilt[fi]

    # ---- image / micrograph names ------------------------------------------- #
    if has_stack2d and "rlnImageName" in df.columns:
        # one MRCS per particle holding exactly the visible tilts, in frame order
        stacks = df["rlnImageName"].to_numpy().astype(str)[pi]
        stacks = np.array([s.split("@", 1)[1] if "@" in s else s for s in stacks])
        out["rlnImageName"] = np.char.add(
            np.char.add((slice_index + 1).astype(str), "@"), stacks)
    elif geom.micrograph_name is not None:
        # no extracted stack: point at the tilt image, to be cropped at rlnCoordinateX/Y
        out["rlnImageName"] = geom.micrograph_name.astype(str)[fi]

    if geom.micrograph_name is not None:
        out["rlnMicrographName"] = geom.micrograph_name.astype(str)[fi]

    for column in keep_columns:
        if column in df.columns:
            out[column] = df[column].to_numpy()[pi]

    return pd.DataFrame(out)
