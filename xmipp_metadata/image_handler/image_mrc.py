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


class ImageMRC(object):
    '''
    Class to read an MRC file
    '''

    DEBUG = False

    def __init__(self, filename=None):
        if filename:
            self.read(filename)
        else:
            self.mrc_handle, self.header = None, None

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
        return self.mrc_handle.data[item]

    def read(self, filename):
        '''
        Reads a given image
            :param filename (str) --> Image to be read
        '''
        try:
            self.mrc_handle = mrcfile.mmap(filename, mode='r+')
            self.header = self.mrc_handle.header
        except ValueError as e:
            print("MRC file header is not valid and raised the following error: ")
            print(e)
            print("We will try to read on permissive mode, and fix the header to recover your data")
            self.mrc_handle = mrcfile.mmap(filename, mode='r+', permissive=True)
            self.mrc_handle.update_header_from_data()
            self.header = self.mrc_handle.header

    def getSamplingRate(self):
        if self.mrc_handle is not None:
            return float(self.mrc_handle.voxel_size.x)
        else:
            return None

    def write(self, data, filename, overwrite=False, sr=1.0):
        sr = 1.0 if sr == 0.0 else sr

        if overwrite and os.path.isfile(filename):
            os.remove(filename)

        with mrcfile.new(filename, overwrite=overwrite) as mrc:
            mrc.set_data(data.astype(np.float32))
            mrc.voxel_size = sr
