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

from skimage.transform import rescale, resize

import mrcfile
from mrcfile.mrcmemmap import MrcMemmap

from .image_spider import ImageSpider


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
            self.binary_file = Path(binary_file)

            if self.binary_file.suffix == ".mrc" or self.binary_file.suffix == ".mrcs":
                self.BINARIES = mrcfile.mmap(self.binary_file, mode='r+')
            elif self.binary_file.suffix == ".stk" or self.binary_file.suffix == ".vol":
                self.BINARIES = ImageSpider(self.binary_file)

    def __getitem__(self, item):
        if isinstance(self.BINARIES, MrcMemmap):
            return self.BINARIES.data[item].copy()
        elif isinstance(self.BINARIES, ImageSpider):
            return self.BINARIES[item].copy()

    def __len__(self):
        if isinstance(self.BINARIES, ImageSpider):
            return len(self.BINARIES)
        elif isinstance(self.BINARIES, MrcMemmap):
            return self.BINARIES.header["nz"]

    def __del__(self):
        self.BINARIES.close()
        print("File closed succesfully!")

    def read(self, binary_file):
        '''
        Reading of a binary image file
            :param binary_file (string) --> Path to the binary file to be read
        '''
        if self.BINARIES:
            self.close()

        self.binary_file = Path(binary_file)

        if self.binary_file.suffix == ".mrc" or self.binary_file.suffix == ".mrcs":
            self.BINARIES = mrcfile.mmap(self.binary_file, mode='r+')
        elif self.binary_file.suffix == ".stk" or self.binary_file.suffix == ".vol":
            self.BINARIES = ImageSpider(self.binary_file)

    def write(self, data, filename=None, overwrite=False):
        if not overwrite and len(self) != data.shape[0] and filename is None:
            raise Exception("Cannot save file. Number of images "
                            "in new data is different. Please, set overwrite to True "
                            "if you are sure you want to do this.")

        filename = self.binary_file if filename is None else Path(filename)

        if filename.suffix == ".mrc" or filename.suffix == ".mrcs":
            with mrcfile.new(filename, overwrite=True) as mrc:
                mrc.set_data(data.astype(np.float32))
        elif filename.suffix == ".stk" or filename.suffix == ".vol":
            self.BINARIES.write(data, filename)

    def convert(self, orig_file, dest_file):
        self.read(orig_file)
        data = self.getData()
        self.write(data, dest_file)

    def getData(self):
        return self[:]

    def getDimensions(self):
        if isinstance(self.BINARIES, ImageSpider):
            return np.asarray([self.BINARIES.header_info["n_slices"],
                               self.BINARIES.header_info["n_rows"],
                               self.BINARIES.header_info["n_columns"]])
        elif isinstance(self.BINARIES, MrcMemmap):
            return np.asarray([self.BINARIES.header["nz"],
                               self.BINARIES.header["ny"],
                               self.BINARIES.header["nx"]])

    def scaleSplines(self, inputFn, outputFn, scaleFactor=None, finalDimension=None,
                     isStack=False):
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
            if isStack:
                aux = []
                for slice in data:
                    aux.append(resize(slice, finalDimension))
                data = np.asarray(aux)
            else:
                data = resize(data, finalDimension)

        self.write(data, outputFn)

    def close(self):
        '''
        Close the current binary file
        '''
        if isinstance(self.BINARIES, ImageSpider):
            self.BINARIES.close()
