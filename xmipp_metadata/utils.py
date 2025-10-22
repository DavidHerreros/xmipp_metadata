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
import math
from emtable import Table
import pandas as pd
from typing import Dict, Union, Optional, Literal
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import affine_transform


# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# RELION → Xmipp mapping (no leading underscore)
RELION_TO_XMIPP_NOUSCORE = {
    # bookkeeping / ids
    "rlnImageId": "itemId",
    "rlnImageName": "image",
    "rlnMicrographName": "micrograph",
    "rlnMicrographId": "micrographId",
    "rlnParticleName": "particleId",
    "rlnGroupName": "groupName",
    "rlnOpticsGroup": "groupId",
    "rlnRandomSubset": "randomSubset",
    "rlnClassNumber": "classNumber",

    # coordinates (particle center on the micrograph)
    "rlnCoordinateX": "xcoor",
    "rlnCoordinateY": "ycoor",
    "rlnCoordinateZ": "zcoor",

    # orientations
    "rlnAngleRot": "angleRot",
    "rlnAngleTilt": "angleTilt",
    "rlnAnglePsi": "anglePsi",

    # shifts (in pixels or Å depending on RELION version)
    "rlnOriginX": "shiftX",
    "rlnOriginY": "shiftY",
    "rlnOriginZ": "shiftZ",
    "rlnOriginXAngst": "shiftX",
    "rlnOriginYAngst": "shiftY",

    # CTF parameters
    "rlnVoltage": "ctfVoltage",
    "rlnSphericalAberration": "ctfSphericalAberration",
    "rlnAmplitudeContrast": "ctfQ0",
    "rlnCtfImage": "ctfImage",
    "rlnCtfBfactor": "ctfBfactor",
    "rlnCtfScalefactor": "ctfScaleFactor",
    "rlnCtfMaxResolution": "ctfMaxResolution",
    "rlnCtfFigureOfMerit": "ctfFom",
    "rlnCtfValue": "ctfValue",
    "rlnDetectorPixelSize": "ctfDetectorPixelSize",
    "rlnMagnification": "ctfMagnification",
    "rlnDefocusU": "ctfDefocusU",
    "rlnDefocusV": "ctfDefocusV",
    "rlnDefocusAngle": "ctfDefocusAngle",
    "rlnCtfDefocusU": "ctfDefocusU",
    "rlnCtfDefocusV": "ctfDefocusV",
    "rlnCtfDefocusAngle": "ctfDefocusAngle",

    # other / quality
    "rlnAutopickFigureOfMerit": "autopickFom",
    "rlnMaxValueProbDistribution": "scoreByVariance",
    "rlnNormCorrection": "normCorrection",
    "rlnLogLikeliContribution": "logLikeContribution",
    "rlnAccuracyRotations": "angleAccuracy",
    "rlnAccuracyTranslations": "shiftAccuracy",
    "rlnNrOfSignificantSamples": "nSamples",
    "rlnReferenceImage": "referenceImage",
}

# Xmipp (no leading underscore) --> RELION labels
# (Complements the earlier RELION->Xmipp mapping you used)
XMIPP_TO_RELION_PARTICLES = {
    # identity / names
    "image": "rlnImageName",
    "micrograph": "rlnMicrographName",
    "micrographId": "rlnMicrographId",
    "itemId": "rlnImageId",
    "particleId": "rlnParticleName",
    "groupName": "rlnGroupName",
    "groupId": "rlnOpticsGroup",     # lives in particles; also used to build optics
    "randomSubset": "rlnRandomSubset",
    "classNumber": "rlnClassNumber",

    # coordinates & geometry
    "xcoor": "rlnCoordinateX",
    "ycoor": "rlnCoordinateY",
    "zcoor": "rlnCoordinateZ",
    "angleRot": "rlnAngleRot",
    "angleTilt": "rlnAngleTilt",
    "anglePsi": "rlnAnglePsi",

    # shifts (note: unit choice handled below)
    # "shiftX" -> rlnOriginX or rlnOriginXAngst
    # "shiftY" -> rlnOriginY or rlnOriginYAngst
    # "shiftZ" -> rlnOriginZ  (Å variant is uncommon in RELION)

    # per-particle CTF / quality
    "ctfImage": "rlnCtfImage",
    "ctfBfactor": "rlnCtfBfactor",
    "ctfScaleFactor": "rlnCtfScalefactor",
    "ctfMaxResolution": "rlnCtfMaxResolution",
    "ctfFom": "rlnCtfFigureOfMerit",
    "ctfValue": "rlnCtfValue",
    "ctfDefocusU": "rlnCtfDefocusU",
    "ctfDefocusV": "rlnCtfDefocusV",
    "ctfDefocusAngle": "rlnCtfDefocusAngle",

    # misc stats
    "autopickFom": "rlnAutopickFigureOfMerit",
    "scoreByVariance": "rlnMaxValueProbDistribution",
    "normCorrection": "rlnNormCorrection",
    "logLikeContribution": "rlnLogLikeliContribution",
    "angleAccuracy": "rlnAccuracyRotations",
    "shiftAccuracy": "rlnAccuracyTranslations",
    "nSamples": "rlnNrOfSignificantSamples",
    "referenceImage": "rlnReferenceImage",
}

# Optics fields to extract from the Xmipp table (group-level)
XMIPP_TO_RELION_OPTICS = {
    "groupId": "rlnOpticsGroup",
    "groupName": "rlnGroupName",  # optional, RELION supports rlnOpticsGroupName (varies by version)
    "ctfVoltage": "rlnVoltage",
    "ctfSphericalAberration": "rlnSphericalAberration",
    "ctfQ0": "rlnAmplitudeContrast",
    "ctfDetectorPixelSize": "rlnDetectorPixelSize",
    "ctfMagnification": "rlnMagnification",
    # Add more group-level fields if you keep them at optics scope in your workflow,
    # e.g. "samplingRate": "rlnImagePixelSize" (if you maintain such a column).
}


def emtable_2_pandas(file_name):
    """Convert an EMTable object to a Pandas dataframe to be used by XmippMetaData class"""

    # Read EMTable
    table = Table(fileName=file_name)

    # Init Pandas table
    pd_table = []

    # Iter rows and set data
    for row in table:
        row = row._asdict()
        for key, value in row.items():
            if isinstance(value, str) and not "@" in value:
                row[key] = value.replace(" ", ",")
        pd_table.append(pd.DataFrame([row]))

    return pd.concat(pd_table, ignore_index=True)


def _choose_particles_table(star_obj: Dict[str, pd.DataFrame]) -> str:
    """
    Heuristic to pick the particles-like table from a starfile.read() dict.
    Prefers any key named 'particles' (case-insensitive), otherwise the table
    containing rlnImageName or rlnMicrographName.
    """
    # 1) direct name hint
    for k in star_obj.keys():
        if k.lower() in {"particles", "data_particles", "particles_table"}:
            return k
    # 2) columns heuristic
    best_key = None
    best_score = -1
    wanted = {"rlnImageName", "rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"}
    for k, df in star_obj.items():
        score = len(wanted.intersection(set(df.columns)))
        if score > best_score:
            best_key = k
            best_score = score
    return best_key or list(star_obj.keys())[0]


def _merge_optics_into_particles(
    particles: pd.DataFrame,
    optics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge optics loop into particles on rlnOpticsGroup if present in both.
    Keeps particle columns when duplicates exist.
    """
    if "rlnOpticsGroup" in particles.columns and "rlnOpticsGroup" in optics.columns:
        # Avoid duplicate columns from optics that already exist in particles
        optics_cols = [c for c in optics.columns if c not in particles.columns or c == "rlnOpticsGroup"]
        merged = particles.merge(
            optics[optics_cols],
            on="rlnOpticsGroup",
            how="left",
            suffixes=("", "_opt")
        )
        return merged
    return particles


def relion_df_to_xmipp_labels(
    star_obj: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    *,
    table: Optional[str] = None,
    merge_optics: bool = True,
    relion_to_xmipp: Dict[str, str] = RELION_TO_XMIPP_NOUSCORE,
    prefer_ctf_defocus_uv: bool = True,
) -> pd.DataFrame:
    """
    Convert a RELION table (or starfile.read() dict) to Xmipp-style labels (no leading underscores).

    Parameters
    ----------
    star_obj : DataFrame or dict[str, DataFrame]
        Either a single RELION table (DataFrame) or the dict returned by starfile.read().
    table : str or None
        If star_obj is a dict, choose which key to use. If None, auto-detect a particles table.
    merge_optics : bool
        If star_obj is a dict and contains 'optics' + particles, merge optics fields into particles.
    relion_to_xmipp : dict
        Mapping from RELION to Xmipp-style column names.
    prefer_ctf_defocus_uv : bool
        When both rlnDefocus* and rlnCtfDefocus* exist, drop the older rlnDefocus* set.

    Returns
    -------
    pd.DataFrame
        DataFrame with Xmipp-style column names (no leading underscore).
    """
    # Pick the working DataFrame
    if isinstance(star_obj, dict):
        if table is None:
            table = _choose_particles_table(star_obj)
        df = star_obj[table].copy()

        # Optionally merge optics into particles
        if merge_optics:
            # try common optics key names
            optics_key = None
            for k in star_obj.keys():
                if k.lower() in {"optics", "data_optics", "optics_table"}:
                    optics_key = k
                    break
            if optics_key is not None and optics_key != table:
                df = _merge_optics_into_particles(df, star_obj[optics_key])
    else:
        df = star_obj.copy()

    # Prefer rlnCtfDefocus* over rlnDefocus* if both present
    if prefer_ctf_defocus_uv:
        pairs = [
            ("rlnDefocusU", "rlnCtfDefocusU"),
            ("rlnDefocusV", "rlnCtfDefocusV"),
            ("rlnDefocusAngle", "rlnCtfDefocusAngle"),
        ]
        for old, new in pairs:
            if old in df.columns and new in df.columns:
                df = df.drop(columns=[old])

    # Apply renaming
    rename_map = {c: relion_to_xmipp[c] for c in df.columns if c in relion_to_xmipp}
    df = df.rename(columns=rename_map)

    return df


def xmipp_df_to_relion_labels(
    xmipp_df: pd.DataFrame,
    *,
    shift_units: Literal["pixels", "angstroms"] = "pixels",
    default_group: int = 1,
    optics_aggregate: Literal["first", "mean", "median"] = "first",
    drop_optics_from_particles: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Convert a single Xmipp-style table (no leading underscores in column names)
    into RELION 'optics' and 'particles' tables with rln* labels.

    Parameters
    ----------
    xmipp_df : pd.DataFrame
        Input table with columns like: image, micrograph, xcoor, ycoor, anglePsi,
        shiftX, ctfVoltage, ctfDefocusU, ..., groupId, etc.
    shift_units : {'pixels','angstroms'}
        Choose how to map shiftX/shiftY:
          - 'pixels'   -> rlnOriginX,  rlnOriginY
          - 'angstroms'-> rlnOriginXAngst, rlnOriginYAngst
        (shiftZ always mapped to rlnOriginZ if present)
    default_group : int
        Used if no 'groupId' is present; all rows are assigned to this group.
    optics_aggregate : {'first','mean','median'}
        How to collapse multiple rows per group into a single optics row
        when optics columns vary within a group.
    drop_optics_from_particles : bool
        After building the optics table, remove optics columns from particles.

    Returns
    -------
    dict with keys 'optics' and 'particles'
    """
    df = xmipp_df.copy()

    # Ensure we have a group id
    if "groupId" not in df.columns:
        df["groupId"] = default_group

    # 1) Build the particles table
    parts = df.copy()

    # Map simple columns
    rename_map_particles = {c: XMIPP_TO_RELION_PARTICLES[c]
                            for c in parts.columns if c in XMIPP_TO_RELION_PARTICLES}

    # Handle shifts by unit
    if "shiftX" in parts.columns:
        rename_map_particles["shiftX"] = "rlnOriginX" if shift_units == "pixels" else "rlnOriginXAngst"
    if "shiftY" in parts.columns:
        rename_map_particles["shiftY"] = "rlnOriginY" if shift_units == "pixels" else "rlnOriginYAngst"
    if "shiftZ" in parts.columns:
        # RELION rarely uses an Å label for Z; map to pixels-style name for compatibility
        rename_map_particles["shiftZ"] = "rlnOriginZ"

    particles = parts.rename(columns=rename_map_particles)

    # 2) Build the optics table (group-level collapse)
    optics_cols_present = [c for c in XMIPP_TO_RELION_OPTICS if c in df.columns]
    if not optics_cols_present:
        # Still need at least the group column
        optics_df = pd.DataFrame({"rlnOpticsGroup": sorted(df["groupId"].unique())})
    else:
        optics_src = df[["groupId"] + [c for c in optics_cols_present if c != "groupId"]].copy()

        # Aggregate to one row per group
        agg_methods = {
            "first": lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
            "mean": "mean",
            "median": "median",
        }
        agg = agg_methods[optics_aggregate]

        grouped = optics_src.groupby("groupId", dropna=False).agg(agg).reset_index()

        # Rename to RELION labels
        rename_map_optics = {c: XMIPP_TO_RELION_OPTICS[c] for c in grouped.columns if c in XMIPP_TO_RELION_OPTICS}
        optics_df = grouped.rename(columns=rename_map_optics)

        # Make sure we have rlnOpticsGroup specifically
        if "rlnOpticsGroup" not in optics_df.columns:
            optics_df = optics_df.rename(columns={"groupId": "rlnOpticsGroup"})
        # RELION often expects integer group ids
        optics_df["rlnOpticsGroup"] = optics_df["rlnOpticsGroup"].astype("int64", errors="ignore")

    # 3) Clean the particles table: ensure rlnOpticsGroup exists & type
    if "rlnOpticsGroup" not in particles.columns and "groupId" in parts.columns:
        particles = particles.rename(columns={"groupId": "rlnOpticsGroup"})
    if "rlnOpticsGroup" in particles.columns:
        try:
            particles["rlnOpticsGroup"] = particles["rlnOpticsGroup"].astype("int64")
        except Exception:
            pass  # leave as-is if conversion fails

    # 4) Optionally drop optics-level columns from particles
    if drop_optics_from_particles:
        to_drop_particle_side = [XMIPP_TO_RELION_OPTICS[c]
                                 for c in optics_cols_present
                                 if c in XMIPP_TO_RELION_OPTICS and XMIPP_TO_RELION_OPTICS[c] in particles.columns]
        # Also drop the Xmipp originals if still around
        to_drop_particle_side += [c for c in optics_cols_present if c in particles.columns]
        to_drop_particle_side = sorted(set(to_drop_particle_side) - {"rlnOpticsGroup", "rlnGroupName"})
        particles = particles.drop(columns=[c for c in to_drop_particle_side if c in particles.columns])

    # 5) Sort columns a bit (optional nicety)
    # Put keys early for readability
    part_key_order = [c for c in ["rlnImageName","rlnImageId","rlnMicrographName","rlnOpticsGroup"]
                      if c in particles.columns]
    particles = particles[[*part_key_order, *[c for c in particles.columns if c not in part_key_order]]]

    opt_key_order = [c for c in ["rlnOpticsGroup","rlnGroupName"] if c in optics_df.columns]
    optics_df = optics_df[[*opt_key_order, *[c for c in optics_df.columns if c not in opt_key_order]]]

    return {"optics": optics_df, "particles": particles}


def fibonacci_sphere(samples):
    """
    Generate points on a unit sphere using the golden ratio-based Fibonacci lattice method.

    Args:
        samples (int): Number of points to generate.

    Returns:
        numpy.ndarray: Array of shape (samples, 3) containing 3D points on the sphere.
    """
    indices = np.arange(0, samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.stack((z, y, x), axis=-1)


def fibonacci_hemisphere(n_points):
    n_points *= 2
    indices = np.arange(0, n_points, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    # Mask to only take the upper hemisphere
    mask = (phi <= np.pi / 2)
    phi = phi[mask]
    theta = theta[mask]

    return theta, phi


def compute_rotations(theta, phi):
    # Rotation about the z-axis by theta
    Rz_theta = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    # Rotation about the y-axis by phi
    Ry_phi = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])
    # Combined rotation matrix
    return Ry_phi @ Rz_theta


# Fourier Slice Interpolator
class FourierInterpolator:
    def __init__(self, volume, pad):
        # Compute the Fourier transform of the volume
        self.size = volume.shape[0]
        self.pad = pad
        volume = np.pad(volume, int(0.25 * self.size * pad))
        self.pad_size = volume.shape[0]
        self.F = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(volume)))
        self.k = np.fft.fftshift(np.fft.fftfreq(volume.shape[0]))
        self.interpolator = RegularGridInterpolator(
            (self.k, self.k, self.k), self.F, bounds_error=False, fill_value=0
        )

    def get_slice(self, rot):
        # Define the grid points in each dimension
        z = np.fft.fftshift(np.fft.fftfreq(self.F.shape[0]))
        y = np.fft.fftshift(np.fft.fftfreq(self.F.shape[1]))
        x = np.fft.fftshift(np.fft.fftfreq(self.F.shape[2]))

        # Define the slice you want to interpolate in Fourier space
        z_slice_index = self.F.shape[0] // 2

        # Create a meshgrid for the slice in Fourier space
        Y, X = np.meshgrid(y, x, indexing='ij')
        Z = np.full_like(X, z[z_slice_index])

        # Flatten the coordinate arrays for transformation
        coords = np.array([X.ravel(), Y.ravel(), Z.ravel()])

        # Rotate the coordinates using the rotation matrix
        rotated_coords = np.dot(rot, coords)
        rotated_coords = np.vstack([rotated_coords[2, :], rotated_coords[1, :], rotated_coords[0, :]])

        # Get projection in real space
        projection = self.interpolator(rotated_coords.T).reshape(Z.shape)
        projection = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(projection))).real

        return projection.copy()


# Real space Slice Interpolator
class RealInterpolator:
    def __init__(self, volume):
        # Compute the Fourier transform of the volume
        self.size = volume.shape[0]
        self.pad_size = volume.shape[0]
        self.volume = volume
        self.k = np.fft.fftshift(np.fft.fftfreq(volume.shape[0]))
        self.interpolator = RegularGridInterpolator(
            (self.k, self.k, self.k), self.volume, bounds_error=False, fill_value=0
        )

    def get_slice(self, rot):
        """
        Rotate and prject a 3D volume using a given rotation matrix around its center.

        Args:
            volume (numpy.ndarray): 3D numpy array representing the volume.
            rotation_matrix (numpy.ndarray): 3x3 rotation matrix.

        Returns:
            numpy.ndarray: 2D projection.
        """
        # Volume shape
        volume_size = self.volume.shape

        # Define the grid points in each dimension
        z = np.fft.fftshift(np.fft.fftfreq(volume_size[0]))
        y = np.fft.fftshift(np.fft.fftfreq(volume_size[1]))
        x = np.fft.fftshift(np.fft.fftfreq(volume_size[2]))

        # Create a meshgrid of coordinates in the Fourier domain
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

        # Flatten the coordinate arrays for transformation
        coords = np.array([X.ravel(), Y.ravel(), Z.ravel()])

        # Rotate the coordinates using the rotation matrix
        rotated_coords = np.dot(rot, coords)

        # Reshape the rotated coordinates back to the original shape
        rotated_Z = rotated_coords[2].reshape(Z.shape)
        rotated_Y = rotated_coords[1].reshape(Y.shape)
        rotated_X = rotated_coords[0].reshape(X.shape)

        # 4. Define the grid interpolator
        interpolator = RegularGridInterpolator((z, y, x), self.volume, method='linear', bounds_error=False,
                                               fill_value=0)

        # Interpolate the Fourier values at the rotated coordinates
        interpolated_values = interpolator((rotated_Z, rotated_Y, rotated_X))

        return np.sum(interpolated_values, axis=0).copy()


# Parallel Projection Computation using Joblib
def compute_projection(rot, interpolator):
    angles = -np.asarray(euler_from_matrix(rot, "szyz"))
    return interpolator.get_slice(np.linalg.inv(rot)), angles


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    >>> angles = (4*math.pi) * (np.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not np.allclose(R0, R1): print(axes, "failed")

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

