"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
from pathlib import Path
from xmipp_metadata import __version__

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
long_description = Path(path.join(here, 'README.md')).read_text()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='xmipp_metadata',  # Required
    version=__version__,  # Required
    description='Package to handle Xmipp Metadata and image binary data',  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',
    url='https://github.com/DavidHerreros/xmipp_metadata',  # Optional
    author='David Herreros',  # Optional
    author_email='dherreros@cnb.csic.es',  # Optional
    keywords='xmipp scipion cryoem imageprocessing scipion-3.0',  # Optional
    packages=find_packages(),
    install_requires=[requirements],
    # entry_points={'pyworkflow.plugin': 'myplugin = myplugin'},
    # package_data={  # Optional
    #    'myplugin': ['icon.png', 'protocols.conf'],
    # }
)
