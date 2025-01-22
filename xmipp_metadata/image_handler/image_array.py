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
import os

import mrcfile


class ImageArray(object):
    '''
    Class to read a Numpy array file
    '''

    DEBUG = False

    def __init__(self, filename=None):
        if isinstance(filename, str):
            self.data = np.load(filename)
        elif isinstance(filename, np.ndarray):
            self.data = filename
        else:
            self.data = None

    def __del__(self):
        '''
        Close the current file before deleting
        '''
        if self.DEBUG:
            print("File closed succesfully!")

    def __len__(self):
        return self.mrc_handle.header["nz"]

    def __iter__(self):
        '''
        Generator method to loop through all the images in the stack
        '''
        for image in self.data:
            yield image

    def __getitem__(self, item):
        return self.data

    def getSamplingRate(self):
        return 1.0

    def write(self, data, filename, overwrite=False, sr=1.0):
        np.save(filename, data)
