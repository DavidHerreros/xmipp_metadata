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


class ImageSpider(object):
    '''
    Class to read an STK image file generated with XMIPP
    '''

    HEADER_OFFSET = 1024
    FLOAT32_BYTES = 4
    TYPE = None

    def __init__(self, filename):
        self.stk_handler = open(filename, "rb")
        self.header_info = self.read_header()
        self.IMG_BYTES = self.FLOAT32_BYTES * self.header_info["img_size"] ** 2

    def __del__(self):
        '''
        Close the current file before deleting
        '''
        self.close()
        print("File closed succesfully!")

    def __len__(self):
        return self.header_info["n_images"]

    def __iter__(self):
        '''
        Generator method to loop through all the images in the stack
        '''
        for iid in range(len(self)):
            yield self.read_image(iid)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.read_image(item)
        elif isinstance(item, list) or isinstance(item, np.ndarray):
            return np.stack([self.read_image(ii) for ii in item])
        elif isinstance(item, slice):
            start = item.start if item.start else 0
            stop = item.stop if item.stop else len(self)
            step = item.step if item.step else 1
            return np.stack([self.read_image(ii) for ii in range(start, stop, step)])

    def read_binary(self, start, end):
        '''
        Read bytes between start and end
            :param start (int) --> Start byte
            :param end (int) --> End byte
            :returns the bytes read
        '''
        self.seek(start)
        return self.stk_handler.read(end)

    def read_numpy(self, start, end):
        '''
        Read bytes between start and end as a Numpy array
            :param start (int) --> Start byte
            :param end (int) --> End byte
            :returns decoded bytes as Numpy array
        '''
        return np.frombuffer(self.read_binary(start, end), dtype=np.float32)

    def seek(self, pos):
        '''
        Move file pointer to a given position
            :param pos (int) --> Byte to move the pointer to
        '''
        self.stk_handler.seek(pos)

    def read_header(self):
        '''
        Reads the header of the current file as a dictionary
            :returns The current header as a dictionary
        '''
        header = self.read_numpy(0, self.HEADER_OFFSET)
        header = header[header != 0.0]
        self.TYPE = "stack" if header[0] == 1.0 else "volume"

        if self.TYPE == "stack":
            header = dict(img_size=int(header[4]), n_images=int(header[10]), offset=int(header[6]))
        else:
            header = dict(img_size=int(header[4]), n_images=int(header[4]), offset=int(header[6]))
        return header

    def read_image(self, iid):
        '''
        Reads a given image in the stack according to its ID
            :param iid (int) --> Image id to be read
            :returns Image as Numpy array
        '''

        if self.TYPE == "stack":
            start = 2 * self.header_info["offset"] + iid * (self.IMG_BYTES + self.header_info["offset"])
        else:
            start = self.header_info["offset"] + iid * self.IMG_BYTES

        img_size = self.header_info["img_size"]
        return self.read_numpy(start, self.IMG_BYTES).reshape([img_size, img_size])

    def close(self):
        '''
        Closes the current file
        '''
        self.stk_handler.close()
