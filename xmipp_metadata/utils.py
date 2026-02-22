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


import shutil
import numpy as np
import math
import pandas as pd
from typing import Dict, Union, Optional, Literal
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R


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


def read_cs_to_relion_df(cs_file_path):
    """
    Reads a CryoSPARC .cs file into a Pandas DataFrame.
    Excludes internal CryoSPARC bookkeeping fields (uid, import_sig).
    """
    data = np.load(cs_file_path)
    names = data.dtype.names
    df_data = {}
    # 1. Track columns we have handled or want to ignore
    converted_cols = []
    # FIELDS TO REMOVE: Add any other non-useful CS internal fields here
    ignored_fields = ['uid', 'blob/import_sig']
    # --- Helper: Rotation Conversion ---
    def convert_pose(pose_data):
        rotations = R.from_rotvec(pose_data)
        return rotations.as_euler('ZYZ', degrees=True)
    # --- 2. Standard Conversions ---
    # Pre-fetch pixel size
    psize = 1.0
    if 'blob/psize_A' in names:
        psize = data['blob/psize_A']
    # OPTICS / CTF
    if 'ctf/defocus_u' in names:
        df_data['rlnDefocusU'] = data['ctf/defocus_u']
        converted_cols.append('ctf/defocus_u')
    if 'ctf/defocus_v' in names:
        df_data['rlnDefocusV'] = data['ctf/defocus_v']
        converted_cols.append('ctf/defocus_v')
    if 'ctf/defocus_angle' in names:
        df_data['rlnDefocusAngle'] = np.degrees(data['ctf/defocus_angle'])
        converted_cols.append('ctf/defocus_angle')
    if 'ctf/accel_kv' in names:
        df_data['rlnVoltage'] = data['ctf/accel_kv']
        converted_cols.append('ctf/accel_kv')
    if 'ctf/cs_mm' in names:
        df_data['rlnSphericalAberration'] = data['ctf/cs_mm']
        converted_cols.append('ctf/cs_mm')
    if 'ctf/amp_contrast' in names:
        df_data['rlnAmplitudeContrast'] = data['ctf/amp_contrast']
        converted_cols.append('ctf/amp_contrast')
    if 'ctf/phase_shift_rad' in names:
        df_data['rlnPhaseShift'] = np.degrees(data['ctf/phase_shift_rad'])
        converted_cols.append('ctf/phase_shift_rad')
    # BLOB / IMAGES
    if 'blob/micrograph_blob/path' in names:
        vals = data['blob/micrograph_blob/path']
        if vals.dtype.kind == 'S': vals = np.char.decode(vals, 'utf-8')
        df_data['rlnMicrographName'] = vals
        converted_cols.append('blob/micrograph_blob/path')
    if 'blob/path' in names:
        vals = data['blob/path']
        if vals.dtype.kind == 'S': vals = np.char.decode(vals, 'utf-8')
        df_data['rlnImageName'] = vals
        converted_cols.append('blob/path')
    if 'blob/idx' in names and 'rlnImageName' in df_data:
        indices = data['blob/idx'] + 1
        df_data['rlnImageName'] = [f"{i}@{p}" for i, p in zip(indices, df_data['rlnImageName'])]
        converted_cols.append('blob/idx')
    if 'blob/psize_A' in names:
        df_data['rlnPixelSize'] = data['blob/psize_A']
        df_data['rlnDetectorPixelSize'] = data['blob/psize_A']
        converted_cols.append('blob/psize_A')
    # ALIGNMENTS
    if 'alignments3D/pose' in names:
        eulers = convert_pose(data['alignments3D/pose'])
        df_data['rlnAngleRot'] = eulers[:, 0]
        df_data['rlnAngleTilt'] = eulers[:, 1]
        df_data['rlnAnglePsi'] = eulers[:, 2]
        converted_cols.append('alignments3D/pose')
    if 'alignments3D/shift' in names:
        shifts = data['alignments3D/shift']
        df_data['rlnOriginXAngst'] = shifts[:, 0] * psize
        df_data['rlnOriginYAngst'] = shifts[:, 1] * psize
        converted_cols.append('alignments3D/shift')
    # --- 3. Robust Catch-All with Exclusion ---
    for name in names:
        # Check if column is already converted OR if it is in our ignore list
        if name not in converted_cols and name not in ignored_fields:
            col_data = data[name]
            # Handle Byte Strings
            if col_data.dtype.kind == 'S':
                col_data = np.char.decode(col_data, 'utf-8')
            # Handle Multi-dimensional Arrays (convert to list of arrays)
            if col_data.ndim > 1:
                df_data[name] = list(col_data)
            else:
                df_data[name] = col_data
    return pd.DataFrame(df_data)


def write_dict_to_cs(input_data, output_filename):
    """
    Converts a Dictionary back into a CryoSPARC .cs file.

    Supports two input formats:
    1. Standard Dict: {'_rlnImageName': [...], '_rlnAngleRot': [...]}
    2. Relion/Xmipp Dict: {'optics': pd.DataFrame, 'particles': pd.DataFrame}
       (Automatically merges optics info into particles)
    """

    # --- 1. INPUT NORMALIZATION ---
    # Convert 'optics'/'particles' structure into a single flat dictionary
    data = {}

    # Check if input is the split format (from xmipp_df_to_relion_labels)
    if isinstance(input_data, dict) and 'particles' in input_data and isinstance(input_data['particles'], pd.DataFrame):
        particles_df = input_data['particles']
        optics_df = input_data.get('optics', None)

        # Merge Optics into Particles if optics exists
        if optics_df is not None and 'rlnOpticsGroup' in particles_df.columns and 'rlnOpticsGroup' in optics_df.columns:
            # Left join particles with optics on Group ID
            merged_df = particles_df.merge(optics_df, on='rlnOpticsGroup', how='left', suffixes=('', '_optics_dup'))

            # Drop duplicate columns resulting from the join (if any)
            cols_to_drop = [c for c in merged_df.columns if c.endswith('_optics_dup')]
            merged_df.drop(columns=cols_to_drop, inplace=True)

            # Convert to dict for the rest of the pipeline
            # We use 'list' to ensure we get python objects, which our validation logic handles well
            temp_data = merged_df.to_dict(orient='list')
        else:
            # No optics or no linking column, just use particles
            temp_data = particles_df.to_dict(orient='list')

        # Normalize keys (ensure _rln prefix) immediately
        for k, v in temp_data.items():
            key_name = k if k.startswith('_') else f"_{k}" if k.startswith('rln') else k
            data[key_name] = v

    else:
        # Standard dictionary input (Legacy support)
        if not input_data:
            raise ValueError("Input dictionary is empty.")
        for k, v in input_data.items():
            key_name = k if k.startswith('_') else f"_{k}" if k.startswith('rln') else k
            data[key_name] = v

    # --- 1b. Determine N (Number of Particles) ---
    priority_keys = ['_rlnImageName', 'blob/path', 'blob/idx', 'uid', 'blob/psize_A']
    ref_key = None
    for pk in priority_keys:
        if pk in data:
            ref_key = pk
            break
    if ref_key is None:
        ref_key = next(iter(data))

    ref_data = np.array(data[ref_key], ndmin=1)
    num_particles = len(ref_data)

    print(f"Detected {num_particles} particles (Reference key: {ref_key})")

    dtype_list = []
    output_arrays = {}
    consumed_keys = set()

    # --- Helper: Get array safely ---
    def get_arr(key, dtype=None):
        return np.array(data[key], dtype=dtype)

    # --- 2. REGENERATE UID ---
    if 'uid' not in data:
        uids = np.random.randint(0, 2 ** 63 - 1, size=num_particles, dtype=np.uint64)
        dtype_list.append(('uid', '<u8'))
        output_arrays['uid'] = uids
    else:
        dtype_list.append(('uid', '<u8'))
        output_arrays['uid'] = get_arr('uid', dtype=np.uint64)
        consumed_keys.add('uid')

    # --- 3. HANDLE STANDARD FIELDS ---

    # Pixel size
    psize = 1.0
    if '_rlnPixelSize' in data:
        psize = get_arr('_rlnPixelSize', dtype=np.float32)
        dtype_list.append(('blob/psize_A', '<f4'))
        output_arrays['blob/psize_A'] = psize
        consumed_keys.add('_rlnPixelSize')
        consumed_keys.add('_rlnDetectorPixelSize')
    elif 'blob/psize_A' in data:
        psize = get_arr('blob/psize_A', dtype=np.float32)

    # 3a. Pose
    if all(k in data for k in ['_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi']):
        eulers = np.stack([
            get_arr('_rlnAngleRot'),
            get_arr('_rlnAngleTilt'),
            get_arr('_rlnAnglePsi')
        ], axis=1)
        rot = R.from_euler('ZYZ', eulers, degrees=True)
        pose_vecs = rot.as_rotvec().astype(np.float32)

        dtype_list.append(('alignments3D/pose', '<f4', (3,)))
        output_arrays['alignments3D/pose'] = pose_vecs
        consumed_keys.update(['_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi'])

    # 3b. Shift
    if '_rlnOriginXAngst' in data:
        shift_x = get_arr('_rlnOriginXAngst')
        shift_y = get_arr('_rlnOriginYAngst')
        scale = 1.0 / psize if np.ndim(psize) == 0 else 1.0 / psize
        shift_vecs = np.stack([shift_x * scale, shift_y * scale], axis=1).astype(np.float32)

        dtype_list.append(('alignments3D/shift', '<f4', (2,)))
        output_arrays['alignments3D/shift'] = shift_vecs
        consumed_keys.update(['_rlnOriginXAngst', '_rlnOriginYAngst'])

    # 3c. Images
    if '_rlnImageName' in data:
        image_strings = get_arr('_rlnImageName').astype(str)

        def parse_relion_path(s):
            if '@' in s:
                idx, p = s.split('@', 1)
                return int(idx) - 1, p
            return 0, s

        parsed = [parse_relion_path(s) for s in image_strings]
        indices = np.array([p[0] for p in parsed], dtype=np.int32)
        paths = np.array([p[1] for p in parsed])

        dtype_list.append(('blob/idx', '<u4'))
        output_arrays['blob/idx'] = indices

        max_len = max(len(p.encode('utf-8')) for p in paths) + 10
        s_type = f'|S{max_len}'
        dtype_list.append(('blob/path', s_type))
        output_arrays['blob/path'] = np.char.encode(paths, 'utf-8')
        consumed_keys.add('_rlnImageName')

    if '_rlnMicrographName' in data:
        paths = get_arr('_rlnMicrographName').astype(str)
        max_len = max(len(p.encode('utf-8')) for p in paths) + 10
        s_type = f'|S{max_len}'
        dtype_list.append(('blob/micrograph_blob/path', s_type))
        output_arrays['blob/micrograph_blob/path'] = np.char.encode(paths, 'utf-8')
        consumed_keys.add('_rlnMicrographName')

    # 3d. CTF
    ctf_map = {
        '_rlnDefocusU': 'ctf/defocus_u',
        '_rlnDefocusV': 'ctf/defocus_v',
        '_rlnVoltage': 'ctf/accel_kv',
        '_rlnSphericalAberration': 'ctf/cs_mm',
        '_rlnAmplitudeContrast': 'ctf/amp_contrast'
    }
    for rln, cs in ctf_map.items():
        if rln in data:
            dtype_list.append((cs, '<f4'))
            output_arrays[cs] = get_arr(rln, dtype=np.float32)
            consumed_keys.add(rln)

    if '_rlnDefocusAngle' in data:
        rads = np.radians(get_arr('_rlnDefocusAngle', dtype=np.float32))
        dtype_list.append(('ctf/defocus_angle', '<f4'))
        output_arrays['ctf/defocus_angle'] = rads
        consumed_keys.add('_rlnDefocusAngle')

    if '_rlnPhaseShift' in data:
        rads = np.radians(get_arr('_rlnPhaseShift', dtype=np.float32))
        dtype_list.append(('ctf/phase_shift_rad', '<f4'))
        output_arrays['ctf/phase_shift_rad'] = rads
        consumed_keys.add('_rlnPhaseShift')

    # --- 4. ROBUST PASS-THROUGH LOGIC ---
    for key in data:
        if key not in consumed_keys and not key.startswith('_rln'):

            raw_vals = data[key]
            final_arr = None

            try:
                test_arr = np.array(list(raw_vals))

                # Validation: Skip garbage columns (header/metadata mismatch)
                is_string = (test_arr.dtype.kind in ('U', 'S')) or \
                            (test_arr.dtype.kind == 'O' and len(test_arr) > 0 and isinstance(raw_vals[0],
                                                                                             (str, np.str_)))

                if len(test_arr) != num_particles:
                    if num_particles == 1 and not is_string:
                        final_arr = test_arr.reshape(1, -1)
                    else:
                        # Warning is suppressed for standard rlnOptics columns that are naturally dropped here
                        continue

                if final_arr is None:
                    if test_arr.dtype.kind == 'O' and not is_string:
                        final_arr = np.vstack(raw_vals).astype(np.float32)
                    elif is_string:
                        s_vals = np.array([str(x) for x in raw_vals])
                        max_len = max(len(s.encode('utf-8')) for s in s_vals) + 10
                        s_type = f'|S{max_len}'
                        dtype_list.append((key, s_type))
                        output_arrays[key] = np.char.encode(s_vals, 'utf-8')
                        continue
                    else:
                        if np.issubdtype(test_arr.dtype, np.integer):
                            final_arr = test_arr.astype(np.int32)
                        else:
                            final_arr = test_arr.astype(np.float32)

                shape = final_arr.shape[1:]
                if np.issubdtype(final_arr.dtype, np.integer):
                    base_type = '<i4'
                else:
                    base_type = '<f4'

                if shape:
                    dtype_list.append((key, base_type, shape))
                else:
                    dtype_list.append((key, base_type))

                output_arrays[key] = final_arr

            except Exception:
                continue

    # --- 5. SAVE ---
    cs_array = np.zeros(num_particles, dtype=dtype_list)
    for name in output_arrays:
        try:
            cs_array[name] = output_arrays[name]
        except Exception:
            pass

    np.save(output_filename, cs_array)
    shutil.move(output_filename + ".npy", output_filename)


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

