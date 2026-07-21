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

from functools import lru_cache

from pathlib import Path

import pandas as pd

import starfile

from xmipp_metadata.image_handler.image_handler import ImageHandler
from xmipp_metadata.utils import emtable_2_pandas, relion_df_to_xmipp_labels, xmipp_df_to_relion_labels, read_cs_to_relion_df, write_dict_to_cs


@lru_cache(maxsize=32)
def _openImageHandler(path, stamp):
    return ImageHandler(path)


def getImageHandler(path):
    '''
    Return an ImageHandler for a stack, reusing an already open one when possible.
    Opening a handler re-parses the file header and remaps the binary, which is
    wasted work when a batch of images is read from the same stack over and over.

    The cache is keyed on the file's mtime and size as well as its path, so a stack
    that has been rewritten is never served stale from the cache.
        :param path (string) --> Path to the binary file
        :returns: an ImageHandler for that file
    '''
    path = str(path)
    try:
        stat = os.stat(path)
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = None  # let ImageHandler raise on a missing file, as before

    return _openImageHandler(path, stamp)


def _isContiguousRun(index):
    '''
    True if index is a run of consecutive ascending integers, so it can be read as a
    plain slice instead of a fancy index.
    '''
    return index.size > 1 and bool(np.all(np.diff(index) == 1))


class XmippMetaData(object):
    '''
    Class to handle and Xmipp MetaData file (and its binaries) in Python

    Parameters:
        :param file_name (string - Optional) --> Path to metadata file
        :param readFrom (string - Optional) --> Can take values:
            - Auto: Determine automatically the best way to read the file
            - Pandas: Read the metadata file as a Pandas table
            - EMTable: Read the metadata file as a EMTable, which will be converted to Pandas later
    '''

    DEBUG = False
    DEFAULT_COLUMN_NAMES = ['anglePsi', 'angleRot', 'angleTilt', 'ctfVoltage', 'ctfDefocusU',
                            'ctfDefocusV', 'ctfDefocusAngle', 'ctfSphericalAberration', 'ctfQ0',
                            'enabled', 'flip', 'image', 'itemId', 'micrograph', 'micrographId',
                            'scoreByVariance', 'scoreByGiniCoeff', 'shiftX', 'shiftY', 'shiftZ',
                            'xcoor', 'ycoor']

    def __init__(self, file_name=None, rows=None, readFrom="Auto", **kwargs):
        # Directory the metadata was read from. Relative image paths in the metadata are
        # defined relative to this directory (not the process CWD), so it is the base used
        # to re-resolve them when writing elsewhere (see ``write(updateImagePaths=True)``).
        # ``None`` for metadata built in-memory (no source file), in which case the CWD is
        # used as a best-effort fallback.
        self._source_dir = None
        if file_name:
            if isinstance(file_name, str):
                if file_name.split(".")[-1] in ["xmd", "star", "cs"]:
                    self.read(file_name, readFrom)
                elif file_name.split(".")[-1] in ["stk", "mrcs"]:  # Create new metadata from images
                    self._source_dir = os.path.dirname(os.path.abspath(file_name))
                    # Fill metadata with images
                    num_images = len(ImageHandler(file_name))
                    angles = kwargs.pop("angles", np.zeros([num_images, 3]))
                    shifts = kwargs.pop("shifts", np.zeros([num_images, 2]))
                    res = {k: v for k, v in kwargs.items() if v is not None}
                    COLUMN_DICT = {'anglePsi': angles[:, 2],
                                   'angleRot': angles[:, 0],
                                   'angleTilt': angles[:, 1],
                                   'enabled': np.ones(num_images, dtype=int),
                                   'image': [f"{id:06d}@{file_name}" for id in np.arange(1, num_images + 1, dtype=int)],
                                   'itemId': np.arange(1, num_images + 1, dtype=int),
                                   'shiftX': shifts[:, 0],
                                   'shiftY': shifts[:, 1],
                                   'shiftZ': np.zeros(num_images),
                                   'ctfVoltage': np.zeros(num_images),
                                   'ctfDefocusU': np.zeros(num_images),
                                   'ctfDefocusV': np.zeros(num_images),
                                   'ctfDefocusAngle': np.zeros(num_images),
                                   'ctfSphericalAberration': np.zeros(num_images)}
                    COLUMN_DICT.update(res)
                    self.table = pd.DataFrame.from_dict(COLUMN_DICT)
                    self.binaries = True
        elif isinstance(rows, list):
            self.table = pd.DataFrame(rows)

            try:
                self.binaries = True
                _ = self.getMetaDataImage(0)
            except (FileNotFoundError, KeyError):
                self.binaries = False

            # Fill non-existing columns
            remain = set(self.DEFAULT_COLUMN_NAMES).difference(set(self.getMetaDataLabels()))
            for label in remain:
                self.table[label] = 0.0
        elif isinstance(rows, pd.DataFrame):
            self.table = rows

            try:
                self.binaries = True
                _ = self.getMetaDataImage(0)
            except (FileNotFoundError, KeyError):
                self.binaries = False

            # Fill non-existing columns
            remain = set(self.DEFAULT_COLUMN_NAMES).difference(set(self.getMetaDataLabels()))
            for label in remain:
                self.table[label] = 0.0
        else:
            self.table = pd.DataFrame(self.DEFAULT_COLUMN_NAMES)
            self.binaries = False

    def __len__(self):
        return self.table.shape[0]

    def __iter__(self):
        '''
        Iter through the rows in the metadata (generator method)
        '''
        for _, row in self.table.iterrows():
            yield row

    def __getitem__(self, item):
        '''
        Slice the metadata. Indexing returns plain values by default -- a Numpy array for
        a row/column selection, and the cell itself for a single entry -- so that
        md[rows, "column"] can be used directly in arithmetic and string handling.

        To get the slice back as a metadata object instead, pass "metadata" as a third
        index element: md[rows, columns, "metadata"].
            :param item --> row, (row, column), or (row, column, flag)
            :returns: Numpy array, a single cell value, or an XmippMetaData
        '''
        if isinstance(item, tuple) and len(item) == 3 and isinstance(item[-1], str):
            row, col, flag = item
        elif isinstance(item, tuple):
            row, col = item
            flag = "numpy"
        else:
            row, col, flag = item, slice(None), "numpy"

        extracted = self.table.loc[row, col]

        if flag != "numpy":
            return XmippMetaData(rows=extracted)

        if hasattr(extracted, "to_numpy"):
            return extracted.to_numpy().copy()

        # A single cell is already a plain value (a string, a float...) and has nothing
        # to convert -- handing it back wrapped would break every caller that uses it
        return extracted

    def __setitem__(self, key, value):
        self.table.loc[key] = value

    def read(self, file_name, readFrom="Auto"):
        '''
        Read a metadata file
            :param file_name (string) --> Path to metadata file
        '''
        # Relative image paths in this file are defined relative to its own directory;
        # remember it so a later write to a different location can re-resolve them.
        self._source_dir = os.path.dirname(os.path.abspath(file_name))

        if readFrom == "Auto":
            try:
                if os.path.splitext(file_name)[1] == ".cs":
                    self.table = read_cs_to_relion_df(file_name)
                else:
                    self.table = starfile.read(file_name)
            except ValueError:
                self.table = emtable_2_pandas(file_name)
        elif readFrom == "Pandas":
            self.table = starfile.read(file_name)
        elif readFrom == "EMTable":
            self.table = emtable_2_pandas(file_name)
        elif readFrom == "CryoSparc":
            self.table = read_cs_to_relion_df(file_name)

        if os.path.splitext(file_name)[1] in [".star", ".cs"]:
            self.table = relion_df_to_xmipp_labels(self.table)

        try:
            self.binaries = True
            _ = self.getMetaDataImage(0)
        except (FileNotFoundError, KeyError):
            self.binaries = False

        # Fill non-existing columns
        remain = set(self.DEFAULT_COLUMN_NAMES).difference(set(self.getMetaDataLabels()))
        for label in remain:
            self.table[label] = 0.0

    def write(self, filename, overwrite=True, updateImagePaths=False):
        '''
        Write current metadata to file
        '''
        # Filename path
        filename_path = Path(filename).resolve().parent

        # Image path
        def composeImageRelPath(image, relative_to):
            # Check if path has the form index@path
            try:
                index, file = image.split("@")
            except ValueError as e:
                index, file = "", image

            # Image absolute path. A relative path is defined relative to the directory the
            # metadata was read from (``_source_dir``), NOT the process CWD -- resolving it
            # against CWD would silently point to the wrong stack whenever the program runs
            # from elsewhere. Fall back to CWD only for in-memory metadata with no source.
            if not os.path.isabs(file):
                base = self._source_dir if self._source_dir is not None else os.getcwd()
                file = os.path.abspath(os.path.join(base, file))

            # Get new relative path
            file = Path(file).resolve()
            file = os.path.relpath(file, start=relative_to)

            # Recompose path
            if index:
                image = index + "@" + file

            return image

        if updateImagePaths:
            for idx in range(len(self)):
                image = self.getMetadataItems(idx, "image")[0]
                image = composeImageRelPath(image, filename_path)
                self.setMetaDataItems(image, idx, "image")

        if os.path.splitext(filename)[1] == ".star":
            table_to_write = xmipp_df_to_relion_labels(self.table)
        elif os.path.splitext(filename)[1] == ".cs":
            table_to_write = xmipp_df_to_relion_labels(self.table)
            write_dict_to_cs(table_to_write, filename)
            return
        else:
            table_to_write = self.table

        starfile.write(table_to_write, filename, overwrite=overwrite)

    def __del__(self):
        '''
        Closes the Metadata file and binaries to save memory
        '''
        if self.DEBUG:
            print("Binaries and MetaData closed successfully!")

    def shape(self):
        '''
        :returns: A tuple with the current metadata shape (rows, columns)
        '''
        return self.table.shape

    def getMetaDataRows(self, idx):
        '''
        Return a set of rows according to idx
            :parameter idx (list - int) --> Indices of the rows to be returned
            :returns The values stored in the desired rows as a Numpy array
        '''
        if isinstance(idx, (list, np.ndarray)) and len(idx) > 1:
            return self.table.iloc[idx].to_numpy().copy()
        else:
            return np.asarray([self.table.iloc[idx]])

    def setMetaDataRows(self, rows, idx):
        '''
        Set new values for metadata rows
        :param rows (Numpy array) --> New data to be set
        :param idx: (list - int) --> Rows indices to be set
        '''
        self.table.loc[idx, :] = rows
        
    def appendMetaDataRows(self, rows):
        self.table.loc[len(self.table.index)] = rows

    def appendMetaData(self, md):
        md.table["itemId"] = len(self) + md.table["itemId"]
        self.table = pd.concat([self.table, md.table], ignore_index=True)

    def getMetadataItems(self, rows_id, columns_id):
        '''
        Returns a slice of data in the metadata
            :param rows_id (list - int) --> Rows ids to be extracted
            :param columns_id (list - string, int) --> Columns names/indices to be extracted
            :return: sliced metadata as Numpy array
        '''
        if isinstance(rows_id, (list, np.ndarray)):
            return self.table.loc[rows_id, columns_id].to_numpy().copy()
        else:
            return np.asarray([self.table.loc[rows_id, columns_id]])

    def setMetaDataItems(self, items, rows_id, columns_id):
        '''
        Set new values for metadata columns
        :param items (Numpy array) --> New data to be set
        :param rows_id (list - int) --> Rows indices to be set
        :param columns_id (list - string, int) --> Columns names/indices to be set
        '''
        self.table.loc[rows_id, columns_id] = items

    def getMetaDataColumns(self, column_names):
        '''
        Return a set of rows according to idx
            :parameter column_names (list - string,int) --> Column names/indices to be returned
            :returns The values stored in the desired columns as a Numpy array
        '''
        return self.table.loc[:, column_names].to_numpy().copy()

    def setMetaDataColumns(self, columns, column_names):
        '''
        Set new values for metadata columns
        :param columns (Numpy array) --> New data to be set
        :param column_names: (list - string,int) --> Columns names/indices to be set
        '''
        self.table.loc[:, column_names] = columns

    def getMetaDataImage(self, row_id, dtype=None):
        '''
        Returns a set of images read from the metadata
            :param row_id (list - int) --> Row indices from where to read the images
            :param dtype (Numpy dtype - Optional) --> Cast the images to this dtype while
                                                      they are read. Fusing the cast into
                                                      the read avoids materialising a
                                                      full-precision copy of the batch
                                                      first, which halves the memory
                                                      traffic when reading a float32 stack
                                                      into, say, float16.
            :returns: Images from metadata as Numpy array (N x Y x X)
        '''
        if not self.binaries:
            print("Binaries not found...")
            return

        images_rows = self.getMetadataItems(row_id, 'image')

        # Group the requested images by the stack holding them, remembering the position
        # each one has to occupy in the output
        stack_id = {}
        stack_order = {}
        for order_id, row in enumerate(images_rows):
            image_id, path = row.split('@') if "@" in row else (row_id, row)
            stack_id.setdefault(path, []).append(int(image_id) - 1)
            stack_order.setdefault(path, []).append(order_id)

        # The output is allocated once and every stack is read straight into its final
        # slots, so the pixels are copied exactly once (with the cast folded in) instead
        # of being stacked and then reordered
        images = None
        for key, values in stack_id.items():
            ih = getImageHandler(key)
            positions = np.asarray(stack_order[key])

            if len(ih) == len(values) == 1:
                block = ih.getData()[None, ...]
            else:
                index = np.asarray(values)

                # Read in file order. A consecutive run collapses to a plain slice, and
                # even a scattered read costs less walked sequentially than jumping
                # back and forth across the stack.
                sorter = np.argsort(index, kind="stable")
                index, positions = index[sorter], positions[sorter]

                if _isContiguousRun(index):
                    block = ih.getBlock(slice(int(index[0]), int(index[-1]) + 1))
                else:
                    block = ih.getBlock(index)

            if images is None:
                images = np.empty((len(images_rows),) + block.shape[1:],
                                  dtype=block.dtype if dtype is None else dtype)

            # getBlock may hand back a view into the memory map; this assignment is what
            # copies it out (and casts it), so nothing aliases the file afterwards
            if _isContiguousRun(positions):
                images[int(positions[0]):int(positions[-1]) + 1] = block
            else:
                images[positions] = block

        return np.squeeze(images)

    def getMetaDataLabels(self):
        '''
        :returns: The metadata labels associated with the column in the current metadata
        '''
        return list(self.table.columns)

    def isMetaDataLabel(self, label):
        '''
        :returns: True or False depending on whether the metadata label is stored in the metadata
        '''
        return label in self.getMetaDataLabels()

    def concatenateMetadata(self, md):
        '''
        Concatenates a metadata file to the current metadata file
        '''
        if isinstance(md, str):
            md = XmippMetaData(md)

        self.table = pd.concat([self.table, md.table])
