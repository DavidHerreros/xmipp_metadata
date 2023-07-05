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


import struct
import numpy as np
from pathlib import Path
import os


class ImageSpider(object):
    '''
    Class to read an STK image file generated with XMIPP
    '''

    HEADER_OFFSET = 1024
    FLOAT32_BYTES = 4
    TYPE = None
    DEBUG = False

    def __init__(self, filename=None):
        if filename:
            self.stk_handler = open(filename, "rb")
            self.header_info = self.read_header()
            self.IMG_BYTES = self.FLOAT32_BYTES * self.header_info["n_columns"] ** 2
        else:
            self.stk_handler, self.header_info, self.IMG_BYTES = None, None, None

    def __del__(self):
        '''
        Close the current file before deleting
        '''
        self.close()
        if self.DEBUG:
            print("File closed succesfully!")

    def __len__(self):
        if self.TYPE == "stack":
            return self.header_info["n_images"]
        elif self.TYPE == "volume":
            return self.header_info["n_slices"]

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

    def read(self, filename):
        '''
        Reads a given image
           :param filename (str) --> Image to be read
        '''
        self.stk_handler = open(filename, "rb")
        self.header_info = self.read_header()
        self.IMG_BYTES = self.FLOAT32_BYTES * self.header_info["n_columns"] ** 2

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

        header = dict(img_size=int(header[1]), n_images=int(header[25]), offset=int(header[21]),
                      n_rows=int(header[1]), n_columns=int(header[11]), n_slices=int(header[0]),
                      sr=float(header[20]))

        self.TYPE = "stack" if header["n_images"] > 1 else "volume"

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

        img_size = self.header_info["n_columns"]
        return self.read_numpy(start, self.IMG_BYTES).reshape([img_size, img_size])

    def write(self, data, filename=None, overwrite=False, sr=1.0):
        data = data.astype(np.float32)
        sr = 1.0 if sr == 0.0 else sr

        if overwrite and os.path.isfile(filename):
            os.remove(filename)

        if filename:
            mode = Path(filename).suffix
            fid = open(filename, "wb")
            if len(data.shape) == 3:
                # Write first header
                header = self.makeSpiderHeader(data, mode=mode, sr=sr)
                fid.writelines(header)

                # Write slices
                for slice in data:
                    if mode == ".stk":
                        header = self.makeSpiderHeader(slice[None, ...], mode=mode, sr=sr)
                        fid.writelines(header)
                    fid.writelines(slice)
            else:
                mode = ".vol"
                # Write image
                data = data[None, ...] if len(data.shape) == 2 else data
                header = self.makeSpiderHeader(data, mode=mode, sr=sr)
                fid.writelines(header)
                fid.writelines(data)
            fid.close()
        else:
            filename = self.stk_handler.name
            mode = Path(filename).suffix
            if data.shape[0] == len(self) or overwrite:
                fid = open(filename, "wb")
                if len(data.shape) == 3:
                    # Write first header
                    header = self.makeSpiderHeader(data, mode=mode, sr=sr)
                    fid.writelines(header)

                    # Write slices
                    for slice in data:
                        if mode == ".stk":
                            header = self.makeSpiderHeader(slice[None, ...], mode=mode, sr=sr)
                            fid.writelines(header)
                        fid.writelines(slice)
                else:
                    mode = ".vol"
                    # Write image
                    data = data[None, ...] if len(data.shape) == 2 else data
                    header = self.makeSpiderHeader(data, mode=mode, sr=sr)
                    fid.writelines(header)
                    fid.writelines(data)
                fid.close()
            else:
                raise Exception("Cannot save file. Number of images "
                                "in new data is different. Please, set overwrite to True "
                                "if you are sure you want to do this.")

    def makeSpiderHeader(self, im, mode, sr=1.0):
        n_slice, nsam, nrow = im.shape
        lenbyt = nsam * 4  # There are labrec records in the header
        labrec = int(1024 / lenbyt)
        if 1024 % lenbyt != 0:
            labrec += 1
        labbyt = labrec * lenbyt
        nvalues = int(labbyt / 4)
        if nvalues < 23:
            return []

        hdr = []
        for i in range(nvalues):
            hdr.append(0.0)

        # NB these are Fortran indices
        hdr[1] = float(n_slice) if mode == ".vol" else 1.0  # nslice (=1 for an image)
        hdr[2] = float(nrow)  # number of rows per slice
        hdr[3] = float(nrow)  # number of records in the image
        hdr[5] = 1.0 if mode == ".stk" else 3.0
        hdr[12] = float(nsam)  # number of pixels per line
        hdr[13] = float(labrec)  # number of records in file header
        hdr[21] = float(sr)  # sampling rate
        hdr[22] = float(labbyt)  # total number of bytes in header
        hdr[23] = float(lenbyt)  # record length in bytes
        hdr[24] = 2.0 if mode == ".stk" else 0.0
        hdr[25] = 1.0 if mode == ".stk" else 0.0
        hdr[26] = float(n_slice) if mode == ".stk" else 1.0  # nobjects (=1 for an image/vol)

        # adjust for Fortran indexing
        hdr = hdr[1:]
        hdr.append(0.0)
        # pack binary data into a string
        return [struct.pack("f", v) for v in hdr]

    def getSamplingRate(self):
        if self.header_info is not None:
            return self.header_info["sr"]
        else:
            return None

    def close(self):
        '''
        Closes the current file
        '''
        if self.stk_handler is not None:
            self.stk_handler.close()
