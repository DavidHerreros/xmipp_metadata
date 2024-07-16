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

from scipy.spatial.transform import Rotation as R
import numpy as np
import time

from xmipp_metadata.image_handler import ImageHandler
from xmipp_metadata.metadata import XmippMetaData


def test_image_handler():
    # Change dir to correct path
    package_path = os.path.abspath(os.path.dirname(__file__))
    data_test_path = os.path.join(package_path, "data")
    os.chdir(data_test_path)


    # Create outputs dir
    if not os.path.isdir("test_outputs"):
        os.mkdir("test_outputs")


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
    ih = ImageHandler("scaled_particles.stk")


    # Get image with ImageHandler
    img = ih[0]


    # Write image (STK)
    ih.write(img, filename=os.path.join("test_outputs", "test.stk"), sr=4.0)


    # Write image (MRC)
    ih.write(img, filename=os.path.join("test_outputs", "test.mrc"), sr=4.0)


    # Raise error due to wrong overwrite
    try:
        ih.write(img, overwrite=False)
    except Exception as e:
        print("Error raised correctly!")


    # Write image stack (STK)
    img = ih[0:10]
    ih.write(img, filename=os.path.join("test_outputs", "test_stack.stk"), sr=4.0)


    # Write volume (VOL)
    ih.write(img, filename=os.path.join("test_outputs", "test.vol"), sr=4.0)


    # Convert (STK to MRCS)
    ih.convert("scaled_particles.stk", os.path.join("test_outputs", "scaled_particles.mrcs"))


    # Convert (STK to EMS)
    ih.convert("scaled_particles.stk", os.path.join("test_outputs", "scaled_particles.ems"))


    # Get dimensions (MRC)
    ih.read(os.path.join("test_outputs", "scaled_particles.mrcs"))
    dims_mrc = ih.getDimensions()


    # Get dimensions (STK)
    ih.read("scaled_particles.stk")
    dims_stk = ih.getDimensions()


    # Get dimensions (EMS)
    ih.read(os.path.join("test_outputs", "scaled_particles.ems"))
    dims_ems = ih.getDimensions()


    # Scale stack (STK)
    ih.scaleSplines("scaled_particles.stk",
                    os.path.join("test_outputs", "test_stack_scaled.stk"),
                    scaleFactor=2.0, isStack=True)


    # Scale image (STK)
    ih.scaleSplines(os.path.join("test_outputs", "test.stk"),
                    os.path.join("test_outputs", "test_scaled.stk"),
                    scaleFactor=2.0)


    # Scale volume (VOL)
    ih.scaleSplines("AK.vol",
                    os.path.join("test_outputs", "test_scaled.vol"),
                    finalDimension=[128, 128, 128])


    # Scale stack (STK) with single int
    ih.scaleSplines("scaled_particles.stk",
                    os.path.join("test_outputs", "test_stack_scaled_int.stk"),
                    finalDimension=128, isStack=True)


    # Scale volume (VOL) with single int
    ih.scaleSplines("AK.vol",
                    os.path.join("test_outputs", "test_scaled_int.vol"),
                    finalDimension=128)


    # Scale volume (MRC) with single int
    ih.scaleSplines("AK.vol",
                    os.path.join("test_outputs", "test_scaled_int.mrc"),
                    finalDimension=128)


    # Scale volume (MRC) with single int (overwrite)
    ImageHandler().scaleSplines(os.path.join("test_outputs", "test_scaled_int.mrc"),
                                os.path.join("test_outputs", "test_scaled_int.mrc"),
                                finalDimension=256, overwrite=True)


    # Scale volume (VOL) with single int (overwrite)
    ImageHandler().scaleSplines(os.path.join("test_outputs", "test_scaled_int.vol"),
                                os.path.join("test_outputs", "test_scaled_int.vol"),
                                finalDimension=256, overwrite=True)


    # Check resize error due to wrong dimensions
    # Raise error due to wrong overwrite
    try:
        ih.scaleSplines("AK.vol",
                        os.path.join("test_outputs", "test_scaled_int.vol"),
                        finalDimension=[128, 128])
    except Exception as e:
        print("Error raised correctly!")


    # Create circular mask (image)
    ih.createCircularMask(os.path.join("test_outputs", "mask_image.mrc"), boxSize=128, is3D=False,
                          sr=4.0)


    # Create circular mask (volume)
    ih.createCircularMask(os.path.join("test_outputs", "mask_vol.mrc"), boxSize=128, is3D=True,
                          sr=4.0)


    # Warp stack (STK) (Rot 90º)
    angle = 0.5 * np.pi
    transform = np.eye(3)
    transform[:-1, :-1] = np.asarray([[np.cos(angle), -np.sin(angle)],
                                      [np.sin(angle), np.cos(angle)]])
    ImageHandler().affineTransform(inputFn="scaled_particles.stk",
                                   outputFn=os.path.join("test_outputs", "test_stack_tr.stk",),
                                   transformation=transform, isStack=True)


    # Warp stack (VOL) (Rot 90º)
    transform = np.eye(4)
    transform[:-1, :-1] = R.from_euler("zyz", [0.0, 0.0, angle]).as_matrix()
    ImageHandler().affineTransform(inputFn="AK.vol",
                                   outputFn=os.path.join("test_outputs", "test_tr.vol"),
                                   transformation=transform, isStack=False)


    # Set sampling rate (VOL)
    ImageHandler().setSamplingRate(os.path.join("test_outputs", "test_tr.vol"), sr=8.0)


    # Add noise (VOL)
    ImageHandler().addNoise(os.path.join("test_outputs", "test_scaled_int.vol"),
                            os.path.join("test_outputs", "test_tr.vol"),
                            avg=0.0, std=0.2, overwrite=True)


    # Generate mask (VOL)
    ih = ImageHandler(os.path.join("test_outputs", "test_tr.vol"))
    start_time = time.time()
    mask = ih.generateMask(iterations=50, boxsize=64, smoothStairEdges=False, dust_size=50)
    end_time = time.time()
    print(end_time - start_time)
    ih.write(mask, os.path.join("test_outputs", "test_generated_mask.vol"), sr=ih.getSamplingRate())
    ih.write(ih.getData() * mask, os.path.join("test_outputs", "test_generated_masked.vol"), sr=ih.getSamplingRate())


    # Generate projections (Fourier)
    ih = ImageHandler(os.path.join("AK.vol"))
    volume = ih.scaleSplines(finalDimension=64)
    print("Generating Fourier projections...")
    start_time = time.time()
    projections, angles = ih.generateProjections(500, degrees=True, pad=0, useFourier=True, volume=volume,
                                                 n_jobs=20)
    end_time = time.time()
    print(end_time - start_time)
    ImageHandler().write(projections, os.path.join("test_outputs", "projections_fourier.stk"))
    ImageHandler().write(volume, os.path.join("test_outputs", "volume_to_poject_fourier.mrc"))
    md = XmippMetaData(os.path.join("test_outputs", "projections_fourier.stk"), angles=angles)
    md.setMetaDataColumns(np.ones(len(md)), "subtomo_labels")
    md.write(os.path.join("test_outputs", "projections_fourier.xmd"), updateImagePaths=True)

    # # Generate projections (Real)
    ih = ImageHandler(os.path.join("AK.vol"))
    volume = ih.scaleSplines(finalDimension=64)
    print("Generating real projections...")
    start_time = time.time()
    projections, angles = ih.generateProjections(500, degrees=True, pad=0, useFourier=False, volume=volume,
                                                 n_jobs=20)
    end_time = time.time()
    print(end_time - start_time)
    ImageHandler().write(projections, os.path.join("test_outputs", "projections_real.stk"))
    ImageHandler().write(volume, os.path.join("test_outputs", "volume_to_poject_real.mrc"))
    md = XmippMetaData(os.path.join("test_outputs", "projections_real.stk"), angles=angles)
    md.setMetaDataColumns(np.ones(len(md)), "subtomo_labels")
    md.write(os.path.join("test_outputs", "projections_real.xmd"), updateImagePaths=True)


if __name__ == '__main__':
    test_image_handler()
