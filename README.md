<h1 align='center'>Xmipp Metadata Handler</h1>

<p align="center">
        
<img alt="Supported Python versions" src="https://img.shields.io/badge/Supported_Python_Versions-3.8_%7C_3.9_%7C_3.10_%7C_3.11_%7C_3.12-blue">
<img alt="GitHub Downloads (all assets, all releases)" src="https://img.shields.io/github/downloads/DavidHerreros/xmipp_metadata/total">
<img alt="GitHub branch check runs" src="https://img.shields.io/github/check-runs/DavidHerreros/xmipp_metadata/master">
<img alt="GitHub License" src="https://img.shields.io/github/license/DavidHerreros/xmipp_metadata">

</p>

<p align="center">
        
<img alt="Xmipp" width="300" src="https://github.com/I2PC/scipion-em-xmipp/raw/devel/xmipp3/xmipp_logo.png">

</p>

This package implements a Xmipp Metadata handling functionality with image binary accession in Python.

# Included functionalities

- **XmippMetadata** class: Reading and writing of Xmipp Metadata files (.xmd)
- **ImageHandler** class: Reading and writing of image binaries stored in the metadata. It support the following formats:
    - MRC files (reading and writing) for stacks and volumes (.mrcs and .mrc)
    - Spider files (reading and writing) for stacks and volumes (.stk and .vol)
    - EM files (reading and writing) for stack and images (.ems and .em)
