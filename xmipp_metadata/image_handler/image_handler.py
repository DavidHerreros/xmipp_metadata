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

from skimage.transform import rescale, resize, warp

from .image_mrc import ImageMRC

from .image_spider import ImageSpider

from .image_em import ImageEM


class ImageHandler(object):
    '''
    Class to open several CryoEM image formats. Currently supported files include:
        - MRC files (supported trough mrcfile package)
        - Xmipp Spider files (STK and VOL)

    Currently, only reading operations are supported
    '''

    BINARIES = None

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

    def write(self, data, filename=None, overwrite=False, sr=1.0):
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

    def convert(self, orig_file, dest_file):
        self.read(orig_file)
        data = self.getData()
        self.write(data, dest_file, sr=self.getSamplingRate())

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

    def scaleSplines(self, inputFn, outputFn, scaleFactor=None, finalDimension=None,
                     isStack=False, overwrite=False):
        self.read(inputFn)
        data = np.squeeze(self.getData())

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

        self.write(data, outputFn, sr=new_sr, overwrite=overwrite)

    def affineTransform(self, inputFn, outputFn, transformation, isStack=False, overwrite=False):
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

        self.write(data, outputFn, sr=sr, overwrite=overwrite)

    def createCircularMask(self, outputFile, boxSize=None, radius=None, center=None, is3D=True,
                           sr=1.0):
        if boxSize is None and radius is None:
            raise ValueError("At least boxSize or radius should be set.")

        boxSize = int(2 * radius) if boxSize is None else boxSize
        radius = 0.5 * boxSize if radius is None else radius
        center = (radius, radius) if center is None else center

        # Create circular mask
        if is3D:
            Z, Y, X = np.ogrid[:boxSize, :boxSize, :boxSize]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[1]) ** 2)
        else:
            Y, X = np.ogrid[:boxSize, :boxSize]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius

        self.write(mask, outputFile, overwrite=True, sr=sr)

    def close(self):
        '''
        Close the current binary file
        '''
        if isinstance(self.BINARIES, ImageSpider):
            self.BINARIES.close()
