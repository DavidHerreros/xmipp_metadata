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


import os
import numpy as np
from emtable import Table
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import affine_transform


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

    return np.stack((x, y, z), axis=-1)


def compute_euler_angles(directions, degrees=True):
    """
    Compute Euler angles (ZYZ convention) for given directions.

    Args:
        directions (numpy.ndarray): Array of shape (N, 3) containing directions.

    Returns:
        numpy.ndarray: Array of shape (N, 3) containing Euler angles in ZYZ format.
    """
    angles = []
    for direction in directions:
        z = np.array([0, 0, 1])
        v = direction
        axis = np.cross(z, v)
        angle = np.arccos(np.dot(z, v) / (np.linalg.norm(z) * np.linalg.norm(v)))
        rot = R.from_rotvec(angle * axis / np.linalg.norm(axis))
        angles.append(rot.as_euler('zyz'))

    if degrees:
        conversion = 180.0 / np.pi
    else:
        conversion = 1.0

    return conversion * np.array(angles)


def rotate_volume(volume, rotation_matrix):
    """
    Rotate a 3D volume using a given rotation matrix around its center.

    Args:
        volume (numpy.ndarray): 3D numpy array representing the volume.
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        numpy.ndarray: Rotated 3D volume.
    """
    # Get the center of the volume
    center = np.array(volume.shape) / 2.0
    # Define the affine transformation matrix
    affine_mat = np.eye(4)
    affine_mat[:3, :3] = rotation_matrix
    affine_mat[:3, 3] = center - rotation_matrix @ center

    # Apply affine transformation
    rotated_volume = affine_transform(volume, affine_mat[:3, :3], offset=affine_mat[:3, 3])
    return rotated_volume
