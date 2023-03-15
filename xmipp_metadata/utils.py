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
import numpy as np
from emtable import Table
import pandas as pd


def emtable_2_pandas(file_name):
    """Convert an EMTable object to a Pandas dataframe to be used by XmippMetaData class"""

    # Read EMTable
    table = Table(fileName=file_name)

    # Init Pandas table
    pd_table = []

    # Iter rows and set data
    for row in table:
        row = row._asdict()
        for key, value in row.items():
            if isinstance(value, str) and not "@" in value:
                value = np.fromstring(value, sep=" ")
                row[key] = ','.join([str(num) for num in value])
        pd_table.append(pd.DataFrame([row]))

    return pd.concat(pd_table, ignore_index=True)
