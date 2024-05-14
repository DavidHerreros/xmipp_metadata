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

    DEBUG = False
    DEFAULT_COLUMN_NAMES = ['anglePsi', 'angleRot', 'angleTilt', 'ctfVoltage', 'ctfDefocusU',
                            'ctfDefocusV', 'ctfDefocusAngle', 'ctfSphericalAberration', 'ctfQ0',
                            'enabled', 'flip', 'image', 'itemId', 'micrograph', 'micrographId',
                            'scoreByVariance', 'scoreByGiniCoeff', 'shiftX', 'shiftY', 'shiftZ',
                            'xcoor', 'ycoor']

    def __init__(self, file_name, readFrom="Auto", **kwargs):
        if file_name:
            if file_name.split(".")[-1] in ["xmd", "star"]:
                self.read(file_name, readFrom)
            elif file_name.split(".")[-1] in ["stk", "mrcs"]:  # Create new metadata from images
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
        extracted = self.table.loc[item]
        if hasattr(extracted, "to_numpy"):
            return extracted.to_numpy().copy()
        else:
            return extracted

    def __setitem__(self, key, value):
        self.table.loc[key] = value

    def read(self, file_name, readFrom="Auto"):
        '''
        Read a metadata file
            :param file_name (string) --> Path to metadata file
        '''
        if readFrom == "Auto":
            try:
                self.table = starfile.read(file_name)
            except ValueError:
                self.table = emtable_2_pandas(file_name)
        elif readFrom == "Pandas":
            self.table = starfile.read(file_name)
        elif readFrom == "EMTable":
            self.table = emtable_2_pandas(file_name)

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

            # Image absolute path
            if not os.path.isabs(file):
                file = os.path.abspath(file)

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
        starfile.write(self.table, filename, overwrite=overwrite)

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

    def getMetaDataImage(self, row_id):
        '''
        Returns a set of images read from the metadata
            :param row_id (list - int) --> Row indices from where to read the images
            :returns: Images from metadata as Numpy array (N x Y x X)
        '''
        if self.binaries:
            images_rows = self.getMetadataItems(row_id, 'image')
            stack_id = {}
            stack_order = {}
            order_id = 0
            for row in images_rows:
                image_id, path = row.split('@') if "@" in row else (row_id, row)
                if path not in stack_id.keys():
                    stack_id[path] = [int(image_id) - 1, ]
                    stack_order[path] = [order_id, ]
                    order_id += 1
                else:
                    stack_id[path].append(int(image_id) - 1)
                    stack_order[path].append(order_id)
                    order_id += 1

            # Read binary file (if needed)
            images = []
            order = []
            for key, values in stack_id.items():
                order.append(np.asarray(stack_order[key]))
                images.append(ImageHandler(key)[values])
            order = np.hstack(order)
            images = np.squeeze(np.vstack(images))

            if order.size > 1:
                # Create an empty array of the same shape as the original array
                reordered_images = np.empty_like(images)

                # Reorder the original array based on the order vector
                reordered_images[order] = images

                return reordered_images
            else:
                return images
        else:
            print("Binaries not found...")

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
