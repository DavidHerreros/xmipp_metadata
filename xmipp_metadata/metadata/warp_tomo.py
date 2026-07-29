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
Warp / M tilt-series particle metadata.

Unlike a RELION-5 particles file, a Warp-1.x / M "particle series" STAR file has
**already been expanded**: there is one row per (particle, tilt image), and
``rlnAngleRot/Tilt/Psi`` plus ``rlnOriginX/YAngst`` are the composed 2D pose of
that tilt image.  Nothing has to be folded in -- which is exactly why CryoDRGN's
``parse_pose_star`` works on these files despite containing no tomography code at
all (``cryodrgn.dataset.TiltSeriesData`` merely groups the rows by
``rlnGroupName``/``rlnGroupNumber``).

So all this module does is what CryoDRGN does, plus what CryoDRGN does not:

  * group the tilt images into particles and emit the dense 1-based
    ``subtomo_labels`` column the tomography networks expect;
  * order the tilts within each particle;
  * carry the dose weighting that Warp/M writes as ``rlnCtfBfactor`` and
    ``rlnCtfScalefactor``, and derive the accumulated dose from it, instead of
    reconstructing it from a ``--dose-per-tilt`` scalar and an assumed
    dose-symmetric scheme.

Note Warp **2.x** (``WarpTools ts_export_particles --2d``) does *not* write this
format: it writes a RELION-5 shaped particles file (``rlnTomoParticleName``,
``rlnTomoVisibleFrames``) plus a matching tomograms STAR file.  Those go through
:mod:`xmipp_metadata.metadata.relion_tomo` instead.
"""

import os
import warnings

import numpy as np
import pandas as pd

from xmipp_metadata.metadata.relion_tomo import _read_star


__all__ = [
    "has_warp_tilt_series_labels",
    "is_warp_tilt_series_star",
    "warp_star_to_tilt_particles",
]


# Warp/M writes the dose weighting as a per-tilt B factor. RELION's convention is
# exp(B * s^2 / 4) with B negative, and Warp sets B = -4 * cumulative_dose * q,
# with q the electron-dose-to-Bfactor constant it uses (4 A^2 per e-/A^2).
_WARP_DOSE_BFACTOR_PER_E = 4.0


_GROUP_LABELS = ("rlnGroupName", "rlnGroupNumber")


def has_warp_tilt_series_labels(cols):
    """
    True when these column names are the ones a Warp-1.x / M tilt-series file uses.

    Necessary but not sufficient: ``rlnCtfScalefactor`` is written by RELION's
    ``CTF::write`` and so also appears in single-particle files, which carry
    ``rlnGroupNumber`` too. Confirm a match with :func:`is_warp_tilt_series_star`.

        :param cols --> iterable of column names
        :returns: bool
    """
    cols = set(cols)
    if "rlnTomoName" in cols and "rlnTomoVisibleFrames" in cols:
        return False            # RELION-5 / Warp 2.x shaped, not this format
    return "rlnCtfScalefactor" in cols and bool(set(_GROUP_LABELS) & cols)


def _groups_are_tilt_series(df, group_label):
    """
    True when a group holds the tilt images of one particle rather than the many
    particles of one micrograph, which is what the same columns mean in SPA.
    """
    grouped = df.groupby(group_label, sort=False)
    sizes = grouped.size()
    if sizes.empty or int(sizes.max()) < 2:
        return False

    # A particle keeps its coordinate across tilts; particles sharing a group do not
    coords = [c for c in ("rlnCoordinateX", "rlnCoordinateY") if c in df.columns]
    if coords:
        moving = grouped[coords].nunique().max(axis=1) > 1
        if float(moving.mean()) > 0.5:
            return False
    return True


def is_warp_tilt_series_star(blocks):
    """
    True when this STAR file is a Warp-1.x / M tilt-series particle file: one row
    per tilt image, already carrying a composed 2D pose.

        :param blocks --> dict of DataFrames, or a single DataFrame
        :returns: bool
    """
    frames = [blocks] if isinstance(blocks, pd.DataFrame) else list(blocks.values())
    for df in frames:
        if not isinstance(df, pd.DataFrame):
            continue
        if "rlnTomoName" in df.columns and "rlnTomoVisibleFrames" in df.columns:
            return False
        if not has_warp_tilt_series_labels(df.columns):
            continue
        label = next(c for c in _GROUP_LABELS if c in df.columns)
        if _groups_are_tilt_series(df, label):
            return True
    return False


def _choose_group_label(df, group_label):
    if group_label is not None:
        if group_label not in df.columns:
            raise ValueError(f"group_label {group_label!r} is not a column of the "
                             f"particles table")
        return group_label
    for candidate in ("rlnGroupName", "rlnGroupNumber", "rlnTomoParticleName",
                      "rlnTomoParticleId"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "no column found to group tilt images into particles; expected one of "
        "rlnGroupName, rlnGroupNumber, rlnTomoParticleName or rlnTomoParticleId. "
        "Pass group_label=... explicitly.")


def warp_star_to_tilt_particles(
        particles_star,
        *,
        group_label=None,
        sort_tilts="scalefactor",
        shift_units="keep",
        dose_from_bfactor=True,
        require_uniform_tilts=False,
):
    """
    Read a Warp-1.x / M tilt-series particle STAR file and attach the grouping and
    ordering information the tomography networks need.

    No geometry is composed: the poses in the file are already per-tilt-image.
    What is added is ``subtomo_labels`` (dense, 1-based, one value per particle),
    ``rlnTomoFrameIndex`` (the tilt's rank inside its particle) and, optionally, a
    dose recovered from ``rlnCtfBfactor``.

    ``sort_tilts`` controls how the tilts of a particle are ordered:

      * ``"scalefactor"`` -- by ``rlnCtfScalefactor`` descending, i.e. from the
        least tilted image outwards.  This is what CryoDRGN does, and for a
        dose-symmetric scheme it recovers acquisition order.
      * ``"bfactor"``     -- by ``rlnCtfBfactor`` descending, i.e. by increasing
        accumulated dose. More directly meaningful when the B factor is present.
      * ``"file"``        -- keep the order the rows appear in.

        :param particles_star (string) --> path to the Warp/M particles STAR file
        :param group_label (string - Optional) --> column identifying the particle a
               tilt image belongs to. Auto-detected when omitted.
        :param sort_tilts (string) --> "scalefactor", "bfactor" or "file"
        :param shift_units (string) --> "keep" leaves the origin columns as found;
               "pixel" rewrites rlnOriginX/YAngst as rlnOriginX/Y in pixels, and
               "angstrom" does the reverse
        :param dose_from_bfactor (bool) --> derive rlnMicrographPreExposure from
               rlnCtfBfactor when the file has no dose column of its own
        :param require_uniform_tilts (bool) --> raise if particles have differing
               numbers of tilt images, instead of just warning
        :returns: pandas DataFrame with RELION labels, one row per tilt image, plus
                  a dense 1-based ``subtomo_labels`` column
    """
    if sort_tilts not in ("scalefactor", "bfactor", "file"):
        raise ValueError(f"sort_tilts must be scalefactor/bfactor/file, "
                         f"got {sort_tilts!r}")
    if shift_units not in ("keep", "pixel", "angstrom"):
        raise ValueError(f"shift_units must be keep/pixel/angstrom, "
                         f"got {shift_units!r}")

    particles_star = os.path.abspath(particles_star)
    blocks = _read_star(particles_star)

    if "particles" in blocks:
        parts = blocks["particles"]
    else:
        parts = max(blocks.values(), key=lambda d: len(d.columns))
    parts = parts.reset_index(drop=True)

    optics = blocks.get("optics")
    if optics is not None and "rlnOpticsGroup" in parts.columns \
            and "rlnOpticsGroup" in optics.columns:
        optics_cols = [c for c in optics.columns
                       if c not in parts.columns or c == "rlnOpticsGroup"]
        parts = parts.merge(optics[optics_cols], on="rlnOpticsGroup", how="left")

    label = _choose_group_label(parts, group_label)

    # Dense 1-based labels, numbered by first appearance so the mapping matches the
    # order CryoDRGN's OrderedDict grouping would produce
    groups = parts[label].to_numpy()
    codes, _ = pd.factorize(pd.Series(groups), sort=False)
    parts = parts.copy()
    parts["subtomo_labels"] = (codes + 1).astype(np.int64)

    counts = np.bincount(codes)
    if counts.size and counts.min() != counts.max():
        message = (f"particles have between {counts.min()} and {counts.max()} tilt "
                   f"images; networks that assume a fixed tilt count will need the "
                   f"set trimmed")
        if require_uniform_tilts:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning)

    # ---- order the tilts inside each particle ------------------------------- #
    if sort_tilts == "file":
        rank_key = None
    elif sort_tilts == "bfactor" and "rlnCtfBfactor" in parts.columns:
        rank_key = parts["rlnCtfBfactor"].to_numpy(dtype=np.float64)
    elif "rlnCtfScalefactor" in parts.columns:
        rank_key = parts["rlnCtfScalefactor"].to_numpy(dtype=np.float64)
        if sort_tilts == "bfactor":
            warnings.warn("sort_tilts='bfactor' but the file has no rlnCtfBfactor; "
                          "falling back to rlnCtfScalefactor", RuntimeWarning)
    else:
        warnings.warn(f"sort_tilts={sort_tilts!r} but the file has neither "
                      f"rlnCtfScalefactor nor rlnCtfBfactor; keeping file order",
                      RuntimeWarning)
        rank_key = None

    if rank_key is None:
        rank = np.concatenate([np.arange(c) for c in counts]) if counts.size else \
            np.zeros(0, dtype=np.int64)
        order = np.argsort(codes, kind="stable")
        frame_index = np.empty(len(parts), dtype=np.int64)
        frame_index[order] = rank
    else:
        # rank descending within each group, without materialising a groupby
        order = np.lexsort((-rank_key, codes))
        frame_index = np.empty(len(parts), dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]) if counts.size else \
            np.zeros(0, dtype=np.int64)
        frame_index[order] = np.arange(len(parts)) - np.repeat(starts, counts)

    parts["rlnTomoFrameIndex"] = frame_index + 1

    # ---- dose ---------------------------------------------------------------- #
    if dose_from_bfactor and "rlnMicrographPreExposure" not in parts.columns \
            and "rlnCtfBfactor" in parts.columns:
        # Warp writes B = -q * dose; invert it so downstream code has a real dose
        parts["rlnMicrographPreExposure"] = (
            -parts["rlnCtfBfactor"].to_numpy(dtype=np.float64)
            / _WARP_DOSE_BFACTOR_PER_E)

    # ---- shift units ---------------------------------------------------------- #
    if shift_units == "pixel" and "rlnOriginXAngst" in parts.columns:
        apix = _pixel_size(parts)
        for angst, px in (("rlnOriginXAngst", "rlnOriginX"),
                          ("rlnOriginYAngst", "rlnOriginY")):
            if angst in parts.columns:
                parts[px] = parts[angst].to_numpy(dtype=np.float64) / apix
                parts = parts.drop(columns=[angst])
    elif shift_units == "angstrom" and "rlnOriginX" in parts.columns:
        apix = _pixel_size(parts)
        for px, angst in (("rlnOriginX", "rlnOriginXAngst"),
                          ("rlnOriginY", "rlnOriginYAngst")):
            if px in parts.columns:
                parts[angst] = parts[px].to_numpy(dtype=np.float64) * apix
                parts = parts.drop(columns=[px])

    return parts


def _pixel_size(df):
    """
    The pixel size the images are on, needed to convert the origins between
    Angstrom and pixels.
    """
    if "rlnImagePixelSize" in df.columns:
        return df["rlnImagePixelSize"].to_numpy(dtype=np.float64)
    if "rlnDetectorPixelSize" in df.columns and "rlnMagnification" in df.columns:
        return (df["rlnDetectorPixelSize"].to_numpy(dtype=np.float64) * 1e4
                / df["rlnMagnification"].to_numpy(dtype=np.float64))
    raise ValueError(
        "cannot convert the origin shifts: the file has neither rlnImagePixelSize "
        "nor rlnDetectorPixelSize + rlnMagnification to derive a pixel size from")
