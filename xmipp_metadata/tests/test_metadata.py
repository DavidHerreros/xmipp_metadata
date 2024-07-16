#!/usr/bin/env python
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


import os
import shutil

from xmipp_metadata.metadata import XmippMetaData


def test_metadata():
    # Change dir to correct path
    package_path = os.path.abspath(os.path.dirname(__file__))
    data_test_path = os.path.join(package_path, "data")
    os.chdir(data_test_path)


    # Clean output tests dir
    for filename in os.listdir("test_outputs"):
        file_path = os.path.join("test_outputs", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


    # Read metadata
    metadata = XmippMetaData("input_particles.xmd")


    # Get image with ImageHandler
    img = metadata.getMetaDataImage(0)


    # Write metadata (do not update paths)
    metadata.write(filename=os.path.join("test_outputs", "test_same_paths.xmd"))


    # Write metadata (update paths)
    metadata.write(filename=os.path.join("test_outputs", "test_new_paths.xmd"), updateImagePaths=True)


    # Merge metadata (duplicate entries)
    metadata.concatenateMetadata(metadata)
    metadata.write(filename=os.path.join("test_outputs", "test_merged_duplicated.xmd"))


if __name__ == '__main__':
    test_metadata()
