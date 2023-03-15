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

from pathlib import Path

import pandas as pd

import starfile

from xmipp_metadata.image_handler.image_handler import ImageHandler
from xmipp_metadata.utils import emtable_2_pandas


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

    DEFAULT_COLUMN_NAMES = ['anglePsi', 'angleRot', 'angleTilt', 'ctfVoltage', 'ctfDefocusU',
                            'ctfDefocusV', 'ctfDefocusAngle', 'ctfSphericalAberration', 'ctfQ0',
                            'enabled', 'flip', 'image', 'itemId', 'micrograph', 'micrographId',
                            'scoreByVariance', 'scoreByGiniCoeff', 'shiftX', 'shiftY', 'shiftZ',
                            'xcoor', 'ycoor']

    def __init__(self, file_name, readFrom="Auto"):
        if file_name:
            if readFrom == "Auto":
                try:
                    self.table = starfile.read(file_name)
                except ValueError:
                    self.table = emtable_2_pandas(file_name)
            elif readFrom == "Pandas":
                self.table = starfile.read(file_name)
            elif readFrom == "EMTable":
                self.table = emtable_2_pandas(file_name)

            binary_file = self.getMetadataItems(0, 'image')
            binary_file = Path(binary_file[0].split('@')[-1])

            # binary_file = binary_file.with_suffix(".mrc")
            # self.binaries = mrcfile.mmap(binary_file, mode='r+')

            self.binaries = ImageHandler(binary_file)

            # Fill non-existing columns
            remain = set(self.DEFAULT_COLUMN_NAMES).difference(set(self.getMetaDataLabels()))
            for label in remain:
                self.table[label] = 0.0

        else:
            self.table = pd.DataFrame(self.DEFAULT_COLUMN_NAMES)
            self.binaries = None

    def __len__(self):
        return self.table.shape[0]

    def __iter__(self):
        '''
        Iter through the rows in the metadata (generator method)
        '''
        for _, row in self.table.iterrows():
            yield row

    def __getitem__(self, item):
        return self.table.loc[item].to_numpy().copy()

    def __setitem__(self, key, value):
        self.table.loc[key] = value

    def read(self, file_name):
        '''
        Read a metadata file
            :param file_name (string) --> Path to metadata file
        '''
        self.table = starfile.read(file_name)
        binary_file = self.getMetadataItems(0, 'image')
        binary_file = Path(binary_file[0].split('@')[-1])

        self.binaries = ImageHandler(binary_file)

        # Fill non-existing columns
        remain = set(self.DEFAULT_COLUMN_NAMES).difference(set(self.getMetaDataLabels()))
        for label in remain:
            self.table[label] = 0.0

    def write(self, file_name, overwrite=False):
        '''
        Write current metadata to file
        '''
        starfile.write(self.table, file_name, overwrite=overwrite)

    def __del__(self):
        '''
        Closes the Metadata file and binaries to save memory
        '''
        self.binaries.close()
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

    def getMetadataItems(self, rows_id, columns_id):
        '''
        Returns a slice of data in the metadata
            :param rows_id (list - int) --> Rows ids to be extracted
            :param columns_id (list - string, int) --> Columns names/indices to be extracted
            :return: sliced metadata as Numpy array
        '''
        if isinstance(rows_id, (list, np.ndarray)) and len(rows_id) > 1:
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

    def getMetaDataImage(self, row_id):
        '''
        Returns a set of images read from the metadata
            :param row_id (list - int) --> Row indices from where to read the images
            :returns: Images from metadata as Numpy array (N x Y x X)
        '''
        stack_id = self.getMetadataItems(row_id, 'image')
        if "@" in stack_id[0]:
            stack_id = [int(path.split('@')[0]) - 1 for path in stack_id]
        else:
            stack_id = row_id

        return self.binaries[stack_id]

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
