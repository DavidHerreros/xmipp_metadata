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


from pathlib import Path

import numpy as np

from joblib import Parallel, delayed
from scipy.ndimage import shift
from scipy.ndimage import binary_dilation, binary_fill_holes

from skimage.transform import rescale, resize, warp
from skimage import filters
from skimage.measure import label
from skimage.morphology import opening, ball

import morphsnakes as ms

from scipy.ndimage import gaussian_filter, median_filter
from scipy.spatial import ConvexHull, Delaunay

from .image_mrc import ImageMRC

from .image_spider import ImageSpider

from .image_em import ImageEM

from xmipp_metadata.utils import fibonacci_hemisphere, compute_rotations, FourierInterpolator, \
    compute_projection, RealInterpolator


class ImageHandler(object):
    '''
    Class to open several CryoEM image formats. Currently supported files include:
        - MRC files (supported trough mrcfile package)
        - Xmipp Spider files (STK and VOL)

    Currently, only reading operations are supported
    '''

    BINARIES = None
    DEBUG = False

    def __init__(self, binary_file=None):
        if binary_file:
            binary_file = str(binary_file)
            if ":mrcs" in binary_file:
                binary_file = binary_file.replace(":mrcs", "")
            elif ":mrc" in binary_file:
                binary_file = binary_file.replace(":mrc", "")

            self.binary_file = Path(binary_file)

            if self.binary_file.suffix == ".mrc" or self.binary_file.suffix == ".mrcs":
                self.BINARIES = ImageMRC(self.binary_file)
            elif self.binary_file.suffix == ".stk" or self.binary_file.suffix == ".vol" \
                    or self.binary_file.suffix == ".xmp" or self.binary_file.suffix == ".spi":
                self.BINARIES = ImageSpider(self.binary_file)
            elif self.binary_file.suffix == ".em" or self.binary_file.suffix == ".ems":
                self.BINARIES = ImageEM(self.binary_file)

    def __getitem__(self, item):
        if isinstance(self.BINARIES, ImageMRC):
            return self.BINARIES[item].copy()
        elif isinstance(self.BINARIES, ImageSpider):
            return self.BINARIES[item].copy()
        elif isinstance(self.BINARIES, ImageEM):
            return self.BINARIES.data[item].copy()
        else:
            return None

    def __len__(self):
        if isinstance(self.BINARIES, ImageSpider):
            return len(self.BINARIES)
        elif isinstance(self.BINARIES, ImageMRC):
            return len(self.BINARIES)
        elif isinstance(self.BINARIES, ImageEM):
            return len(self.BINARIES)
        else:
            return 0

    def __del__(self):
        if self.BINARIES is not None and isinstance(self.BINARIES, ImageSpider):
            self.BINARIES.close()
        if self.DEBUG:
            print("File closed succesfully!")

    def read(self, binary_file):
        '''
        Reading of a binary image file
            :param binary_file (string) --> Path to the binary file to be read
        '''
        if self.BINARIES:
            self.close()

        binary_file = str(binary_file)
        if ":mrcs" in binary_file:
            binary_file = binary_file.replace(":mrcs", "")
        elif ":mrc" in binary_file:
            binary_file = binary_file.replace(":mrc", "")

        self.binary_file = Path(binary_file)

        if self.binary_file.suffix == ".mrc" or self.binary_file.suffix == ".mrcs":
            self.BINARIES = ImageMRC(self.binary_file)
        elif self.binary_file.suffix == ".stk" or self.binary_file.suffix == ".vol" \
                or self.binary_file.suffix == ".xmp" or self.binary_file.suffix == ".spi":
            self.BINARIES = ImageSpider(self.binary_file)
        elif self.binary_file.suffix == ".em" or self.binary_file.suffix == ".ems":
            self.BINARIES = ImageEM(self.binary_file)

        return self

    def write(self, data, filename=None, overwrite=True, sr=1.0):
        if not overwrite and filename is None and len(self) != data.shape[0]:
            raise Exception("Cannot save file. Number of images "
                            "in new data is different. Please, set overwrite to True "
                            "if you are sure you want to do this.")

        filename = self.binary_file if filename is None else Path(filename)
        sr = 1.0 if sr == 0.0 else sr

        if filename.suffix == ".mrc" or filename.suffix == ".mrcs":
            ImageMRC().write(data, filename, overwrite=overwrite, sr=sr)
        elif filename.suffix == ".stk" or filename.suffix == ".vol" \
                or filename.suffix == ".xmp" or filename.suffix == ".spi":
            ImageSpider().write(data, filename, overwrite=overwrite, sr=sr)
        elif filename.suffix == ".em" or filename.suffix == ".ems":
            ImageEM().write(data, filename, overwrite=overwrite, sr=sr)

    def convert(self, orig_file, dest_file, overwrite=True):
        self.read(orig_file)
        data = self.getData()
        self.write(data, dest_file, sr=self.getSamplingRate(), overwrite=overwrite)

    def getData(self):
        return self[:]

    def getDimensions(self):
        if isinstance(self.BINARIES, ImageSpider):
            return np.asarray([self.BINARIES.header_info["n_slices"],
                               self.BINARIES.header_info["n_rows"],
                               self.BINARIES.header_info["n_columns"]])
        elif isinstance(self.BINARIES, ImageMRC):
            return np.asarray([self.BINARIES.header["nz"],
                               self.BINARIES.header["ny"],
                               self.BINARIES.header["nx"]])
        elif isinstance(self.BINARIES, ImageEM):
            return np.asarray([self.BINARIES.header["zdim"],
                               self.BINARIES.header["ydim"],
                               self.BINARIES.header["xdim"]])

    def getSamplingRate(self):
        if self.BINARIES is not None:
            return self.BINARIES.getSamplingRate()
        else:
            return None

    def setSamplingRate(self, input_file, sr):
        self.read(input_file)
        self.write(np.squeeze(self.getData()), sr=sr, overwrite=True)

    def scaleSplines(self, inputFn=None, outputFn=None, scaleFactor=None, finalDimension=None,
                     isStack=False, overwrite=True, data=None):
        if isinstance(inputFn, str):
            self.read(inputFn)
            data = np.squeeze(self.getData())
        elif self.BINARIES is not None:
            data = np.squeeze(self.getData())
        elif not isinstance(data, np.ndarray):
            raise ValueError('Data to be scaled not found. Please, provide one of the following:'
                             '    - inputFn (str): Path to the file binaries to be scaled'
                             '    - data (np.ndarray): Data to be scaled'
                             '    - Use the read method of this class with a file (example: ImageHandler().read("file")')

        if finalDimension is None:
            if isStack:
                aux = []
                for slice in data:
                    aux.append(rescale(slice, scaleFactor))
                data = np.asarray(aux)
            else:
                data = rescale(data, scaleFactor)
        else:
            # First check if dimesions are ok
            if isinstance(finalDimension, list) and len(finalDimension) != len(data.shape):
                raise ValueError(f"Resize dimensions do not match. You provided "
                                 f"{len(finalDimension)} dimensions, but data has "
                                 f"{len(data.shape)} dimensions")

            if isStack:
                aux = []
                if isinstance(finalDimension, int):
                    finalDimension = finalDimension * np.ones(len(data[0].shape))
                for slice in data:
                    aux.append(resize(slice, finalDimension))
                data = np.asarray(aux)
            else:
                if isinstance(finalDimension, int):
                    finalDimension = finalDimension * np.ones(len(data.shape))
                data = resize(data, finalDimension)

        scaleFactor = finalDimension[0] / data.shape[0] if scaleFactor is None else scaleFactor
        new_sr = self.getSamplingRate() / scaleFactor
        new_sr = new_sr if new_sr > 0.0 else 1.0

        if isinstance(outputFn, str):
            self.write(data, outputFn, sr=new_sr, overwrite=overwrite)
        else:
            return data

    def affineTransform(self, transformation, inputFn=None, outputFn=None, isStack=False, overwrite=True):
        if isinstance(inputFn, str):
            self.read(inputFn)
        data = np.squeeze(self.getData())

        # Transformation dim
        tr_dim = data.ndim if isStack else data.ndim + 1

        # Make transformation homogeneous (if not already)
        if transformation.shape[0] != tr_dim:
            transformation, aux = np.eye(tr_dim), transformation
            transformation[:-1, :-1] = aux

        # Get inverse transform
        inv_transformation = np.eye(tr_dim)
        inv_transformation[:-1, :-1] = np.linalg.inv(transformation[:-1, :-1])
        inv_transformation[:-1, -1] = -inv_transformation[:-1, :-1] @ transformation[:-1, -1]

        # Compute offset to rotate around image centre
        offset = 0.5 * np.asarray(data.shape[1:]) if isStack else 0.5 * np.asarray(data.shape)

        # Get grid coords
        if isStack:
            shape = [data.shape[1], data.shape[2]]
            Y, X = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
            coords = np.array([X.ravel(), Y.ravel()])
        elif data.ndim == 2:
            shape = data.shape
            Y, X = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
            coords = np.array([X.ravel(), Y.ravel()])
        else:
            shape = data.shape
            Z, Y, X = np.meshgrid(np.arange(shape[0]),
                                  np.arange(shape[1]),
                                  np.arange(shape[2]))
            coords = np.array([X.ravel(), Y.ravel(), Z.ravel()])
        coords = coords - offset[:, None]

        # Homogeneous coords
        coords = np.vstack([coords, np.zeros([1, coords.shape[1]])])

        # Get rotated coords
        rot_coords = inv_transformation @ coords

        # Coords in ZYX
        rot_coords = rot_coords[:-1, :]

        # Undo offset
        rot_coords = rot_coords + offset[:, None]

        # Warp coords
        coords = rot_coords.reshape([-1, ] + list(shape))

        if isStack:
            aux = []
            for slice in data:
                aux.append(warp(slice, coords))
            data = np.asarray(aux)
        else:
            data = warp(data, coords)

        sr = self.getSamplingRate() if self.getSamplingRate() > 0.0 else 1.0

        if isinstance(outputFn, str):
            self.write(data, outputFn, sr=sr, overwrite=overwrite)
        else:
            return data

    def createCircularMask(self, outputFile=None, boxSize=None, radius=None, center=None, is3D=True,
                           sr=1.0):
        if boxSize is None and radius is None:
            raise ValueError("At least boxSize or radius should be set.")

        boxSize = int(2 * radius) if boxSize is None else boxSize
        radius = 0.5 * boxSize if radius is None else radius

        if boxSize is not None:
            center = (0.5 * boxSize, 0.5 * boxSize)
        elif radius is not None:
            center = (radius, radius)

        # Create circular mask
        if is3D:
            Z, Y, X = np.ogrid[:boxSize, :boxSize, :boxSize]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[1]) ** 2)
        else:
            Y, X = np.ogrid[:boxSize, :boxSize]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius

        if isinstance(outputFile, str):
            self.write(mask, outputFile, overwrite=True, sr=sr)
        else:
            return mask

    def addNoise(self, input_file=None, output_file=None, std=1.0, avg=0.0, overwrite=True):
        if isinstance(input_file, str):
            self.read(input_file)
        data = np.squeeze(self.getData())
        noise = np.random.normal(loc=avg, scale=std, size=data.shape)
        data_noise = data + noise
        if isinstance(output_file, str):
            self.write(data_noise, output_file, overwrite=overwrite, sr=self.getSamplingRate())
        else:
            return output_file

    def __fillHull(self, image):
        """
        Compute the convex hull of the given binary image and
        return a mask of the filled hull.

        Adapted from:
        https://stackoverflow.com/a/46314485/162094
        This version is slightly (~40%) faster for 3D volumes,
        by being a little more stingy with RAM.
        """
        # (The variable names below assume 3D input,
        # but this would still work in 4D, etc.)

        assert (np.array(image.shape) <= np.iinfo(np.int16).max).all(), \
            f"This function assumes your image is smaller than {2 ** 15} in each dimension"

        points = np.argwhere(image).astype(np.int16)
        hull = ConvexHull(points)
        deln = Delaunay(points[hull.vertices])

        # Instead of allocating a giant array for all indices in the volume,
        # just iterate over the slices one at a time.
        idx_2d = np.indices(image.shape[1:], np.int16)
        idx_2d = np.moveaxis(idx_2d, 0, -1)

        idx_3d = np.zeros((*image.shape[1:], image.ndim), np.int16)
        idx_3d[:, :, 1:] = idx_2d

        mask = np.zeros_like(image, dtype=bool)
        for z in range(len(image)):
            idx_3d[:, :, 0] = z
            s = deln.find_simplex(idx_3d)
            mask[z, (s != -1)] = 1

        return mask

    def generateMask(self, inputFn=None, outputFn=None, iterations=150, smoothing=0, lambda1=1, lambda2=2, std=1,
                     boxsize=128, smoothStairEdges=False, keep_largest=True, dust_size=None, convex=True,
                     applyThreshold=True, threshold="otsu"):
        '''Generate automatically a binary protein mask based on a combination of snakes and
        Otsu method.
            :param int iterations: Number of iterations for computing the snake mask.
            :param int smoothing: Number of times the smoothing operator is applied per iteration.
                                  Reasonable values are around 0-4. Larger values lead to smoother
                                  segmentations.
            :param int lambda1: Weight parameter for the outer region. If `lambda1` is larger than
                                `lambda2`, the outer region will contain a larger range of values than
                                the inner region.
            :param int lambda2: Weight parameter for the inner region. If `lambda2` is larger than
                                `lambda1`, the inner region will contain a larger range of values than
                                the outer region.
            :param int std: Standard deviation to prefilter the map based on a Gaussian filter. Useful when
                            dealing with noisy maps
            :param int boxsize: Compute the snake mask for a map downsampled/upscaled to this size. We recommend
                                to downsample the map to a boxsize between 64px - 128px to improve performance.
                                The output size of the mask generated will keep the original map dimensions, even
                                if this parameter is used
            :param bool smoothStairEdges: Smooth the snake mask borders to avoid stair caise like borders.
                                          We recommend setting it to True if boxsized downsamples the original map.
            :param bool keep_largest: Keep the largest component only detected by the snakes mask. This will help the
                                      posterior Otsu thresholding step to find the appropriate threshold to segment the
                                      protein.
            :param str threshold: Threshold method use to improve the snakes masking. Valid options are:
                                  ["isodata", "li", "mean", "minimum", "otsu", "triangle", "yen"]
            :param float dust_size: Removes any connected component whose size is smaller than or equal to this value
            :param bool convex: Determines whether to create convex mask from the estimated morphSnake mask before
                                further processing it
            :param bool applyThreshold: Determines whether to apply a final thresholding step to generate a tight mask
                                        to the input image/volume
        '''
        # Read the data
        if inputFn is not None:
            data = ImageHandler().read(inputFn).getData()
        else:
            data = np.squeeze(self.getData())
        data_ori = data.copy()

        # Filter to remove noise (optional step)
        if std is not None:
            data = gaussian_filter(data, std)

        # Downscale to reduce execution times (optional but highly recommended)
        if boxsize is not None:
            ori_boxsize = data.shape[0]
            finalDimension = boxsize * np.ones(len(data.shape))
            data = resize(data, finalDimension)

        # Initialization for snakes
        init_ls = ms.circle_level_set(data.shape)

        # Generate snake mask
        acwe_ls1 = ms.morphological_chan_vese(data, iterations=iterations,
                                              init_level_set=init_ls, smoothing=smoothing,
                                              lambda1=lambda1, lambda2=lambda2)

        # Keep the largest component only
        if keep_largest:
            labels = label(acwe_ls1)
            assert (labels.max() != 0)  # assume at least 1 CC
            acwe_ls1 = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

        # Upscale mask (if downscaling was applied)
        if boxsize is not None:
            finalDimension = ori_boxsize * np.ones(len(data.shape))
            acwe_ls1 = resize(acwe_ls1.astype(bool), finalDimension).astype(np.float32)

        # Remove dust
        if dust_size is not None:
            acwe_ls1 = opening(acwe_ls1.astype(bool), ball(5))
            labels = label(acwe_ls1)
            assert (labels.max() != 0)  # assume at least 1 CC
            keep_regions = (np.argwhere(np.bincount(labels.flat)[1:] > dust_size) + 1)
            aux = np.zeros(acwe_ls1.shape)
            for lid in keep_regions:
                aux += labels == lid
            acwe_ls1 = aux.astype(np.float32)

        if smoothStairEdges:
            acwe_ls1 = (median_filter(acwe_ls1, size=5) >= 0.001).astype(np.float32)
        
        if convex:
            acwe_ls1 = self.__fillHull(acwe_ls1.astype(bool)).astype(float)

        if applyThreshold:
            data_ori = data_ori * acwe_ls1
            threshold_fun = getattr(filters, "threshold_" + threshold)
            acwe_ls1 = (data_ori >= threshold_fun(data_ori)).astype(np.float32)

        if outputFn is not None:
            ImageHandler().write(acwe_ls1, outputFn)
        else:
            return acwe_ls1

    def floodFillMask(self, data, outputFn=None):
        ball_kernel = ball(2)
        for _ in range(10):
            data = binary_dilation(data, ball_kernel)
        data = binary_fill_holes(data, ball_kernel)

        if outputFn is not None:
            ImageHandler().write(data, outputFn)
        else:
            return data

    def generateProjections(self, num_projections, degrees=True, volume=None, n_jobs=8, pad=0, useFourier=True):
        """
        Generate 2D projections of a 3D volume uniformly over the projection sphere.

        Args:
            volume (numpy.ndarray): 3D numpy array representing the volume.
            num_projections (int): Number of projections to generate.

        Returns:
            projections (list of numpy.ndarray): List containing 2D projections.
            euler_angles (numpy.ndarray): Array of shape (num_projections, 3) containing
                                          Euler angles in ZYZ format for each projection.
        """
        # Read the data
        if volume is None:
            volume = np.squeeze(self.getData())

        theta, phi = fibonacci_hemisphere(num_projections)
        rots = np.stack([compute_rotations(t, p) for t, p in zip(theta, phi)], axis=0)

        if useFourier:
            interpolator = FourierInterpolator(volume, pad=pad)
        else:
            interpolator = RealInterpolator(volume)

        results = np.array(Parallel(n_jobs=n_jobs)(delayed(compute_projection)(rot, interpolator) for rot in rots),
                           dtype=object)
        projections, euler_angles = np.stack(results[:, 0], axis=0), np.stack(results[:, 1], axis=0)

        if degrees:
            euler_angles = np.rad2deg(euler_angles)

        return projections, euler_angles


    def close(self):
        '''
        Close the current binary file
        '''
        if isinstance(self.BINARIES, ImageSpider):
            self.BINARIES.close()
