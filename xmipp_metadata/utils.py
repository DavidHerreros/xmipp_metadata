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
from emtable import Table
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
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

    return np.stack((z, y, x), axis=-1)


def compute_rotations(directions):
    """
    Compute Euler angles (ZYZ convention) for given directions.

    Args:
        directions (numpy.ndarray): Array of shape (N, 3) containing directions.

    Returns:
        numpy.ndarray: Array of shape (N, 3) containing Euler angles in ZYZ format.
    """
    rots = []
    for direction in directions:
        z = np.array([1, 0, 0])
        v = direction
        axis = np.cross(z, v)
        angle = np.arccos(np.dot(z, v) / (np.linalg.norm(z) * np.linalg.norm(v)))
        if np.linalg.norm(axis) < 1e-6:
            rot = R.from_euler('xyx', [0, 0, 0])
        else:
            rot = R.from_rotvec(angle * axis / np.linalg.norm(axis))
        rots.append(rot.as_matrix())
    return np.stack(rots, axis=0)


def rotate_project_volume(volume, rotation_matrix):
    """
    Rotate and prject a 3D volume using a given rotation matrix around its center.

    Args:
        volume (numpy.ndarray): 3D numpy array representing the volume.
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        numpy.ndarray: 2D projection.
    """
    # Get the center of the volume
    center = np.array(volume.shape) / 2.0
    # Define the affine transformation matrix
    affine_mat = np.eye(4)
    affine_mat[:3, :3] = rotation_matrix
    affine_mat[:3, 3] = center - rotation_matrix @ center

    # Apply affine transformation
    rotated_volume = affine_transform(volume, affine_mat[:3, :3], offset=affine_mat[:3, 3])
    angles = R.from_matrix(rotation_matrix).as_euler("xyx")
    return np.sum(rotated_volume, axis=0), angles

# Fourier Slice Interpolator
class FourierInterpolator:
    def __init__(self, volume, pad):
        # Compute the Fourier transform of the volume
        self.size = volume.shape[0]
        self.pad = pad
        volume = np.pad(volume, int(0.25 * self.size * pad))
        self.pad_size = volume.shape[0]
        F = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(volume)))
        self.k = np.linspace(-self.pad_size / 2, self.pad_size / 2, self.pad_size)
        self.interpolator = RegularGridInterpolator(
            (self.k, self.k, self.k), F, bounds_error=False, fill_value=0
        )

    def get_slice(self, rot):
        # Create the grid for the desired slice
        x = np.linspace(-self.pad_size / 2, self.pad_size / 2, self.size)
        y = np.linspace(-self.pad_size / 2, self.pad_size / 2, self.size)
        xx, yy = np.meshgrid(x, y, indexing='xy')

        # Apply the rotation matrix to obtain Fourier coordinates
        kx = rot[0, 0] * np.zeros_like(xx) + rot[0, 1] * yy + rot[0, 2] * xx
        ky = rot[1, 0] * np.zeros_like(xx) + rot[1, 1] * yy + rot[1, 2] * xx
        kz = rot[2, 0] * np.zeros_like(xx) + rot[2, 1] * yy + rot[2, 2] * xx

        coords = np.stack([kx, ky, kz], axis=-1)
        return (np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(self.interpolator(coords))))) ** 2).copy()


# Parallel Projection Computation using Joblib
def compute_projection(rot, interpolator):
    angles = R.from_matrix(rot).as_euler("xyx")
    return interpolator.get_slice(rot), angles
