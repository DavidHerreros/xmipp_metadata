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
            binary_file = Path(binary_file)

            if binary_file.suffix == ".mrc":
                self.BINARIES = mrcfile.mmap(binary_file, mode='r+')
            elif binary_file.suffix == ".stk" or binary_file.suffix == ".vol":
                self.BINARIES = ImageSpider(binary_file)

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

        if binary_file.suffix == ".mrc":
            self.BINARIES = mrcfile.mmap(binary_file, mode='r+')
        elif binary_file.suffix == ".stk":
            self.BINARIES = ImageSpider(binary_file)

    def close(self):
        '''
        Close the current binary file
        '''
        if isinstance(self.BINARIES, ImageSpider):
            self.BINARIES.close()
