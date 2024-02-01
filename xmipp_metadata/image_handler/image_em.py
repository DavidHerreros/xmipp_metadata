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

import emfile


class ImageEM(object):
    '''
    Class to read an EM file
    '''

    DEBUG = False

    def __init__(self, filename=None):
        if filename:
            self.read(filename)
        else:
            self.header, self.data = None, None

    def __del__(self):
        '''
        Close the current file before deleting
        '''
        if self.DEBUG:
            print("File closed succesfully!")

    def __len__(self):
        return self.header["zdim"]

    def __iter__(self):
        '''
        Generator method to loop through all the images in the stack
        '''
        for image in self.data:
            yield image

    def __getitem__(self, item):
        return self.data[item]

    def read(self, filename):
        '''
        Reads a given image
            :param filename (str) --> Image to be read
        '''
        self.header, self.data = emfile.read(filename, mmap=True, header_only=False)

    def getSamplingRate(self):
        if self.header is not None:
            return self.header["SPx"]
        else:
            return None

    def write(self, data, filename, overwrite=False, sr=1.0):
        sr = 1.0 if sr == 0.0 else sr
        header_params = {"SPx": sr, "SPy": sr, "SPz": sr}

        if overwrite and os.path.isfile(filename):
            os.remove(filename)

        emfile.write(filename, data.astype(np.float32), header_params=header_params,
                     overwrite=overwrite)
