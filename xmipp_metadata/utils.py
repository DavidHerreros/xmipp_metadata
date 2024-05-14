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

