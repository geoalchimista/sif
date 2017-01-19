"""
Read satellite data of solar induced fluorescence
- `read_gome2_l2()`: read GOME-2 SIF Level 2 data

"""
import os
import numpy as np
import pandas as pd
import netCDF4


def read_gome2_l2(filepath, lat=None, lon=None):
    """
    Read GOME-2 SIF Level 2 data (whole file or single point).

    (c) 2017 Wu Sun <wu.sun@ucla.edu>

    Parameters
    ----------
    @TODO: to be completed

    Return
    ------
    df_sif : pandas.DataFrame


    Raise
    -----
    ValueError

    """
    nc_fid = netCDF4.Dataset(filepath, 'r')

    variable_names = [key for key in nc_fid.variables if key not in
                      ['time', 'Calibration_factor',
                       'Latitude_corners', 'Longitude_corners']]
    df_sif = pd.DataFrame(columns=variable_names)

    for var in variable_names:
        df_sif[var] = nc_fid.variables[var][:]

    df_sif.insert(0, 'datetime', np.datetime64('nat'))

    date_str = os.path.basename(filepath).split('nolog.')[1][0:8]
    date_str = '-'.join([date_str[0:4], date_str[4:6], date_str[6:8]])
    for i in range(df_sif.shape[0]):
        time_str = ''.join(nc_fid.variables['time'][i].astype('|U1'))
        df_sif = df_sif.set_value(
            i, 'datetime', np.datetime64('T'.join([date_str, time_str])))

    df_sif.loc[df_sif['datetime'] < df_sif.loc[0, 'datetime'], 'datetime'] += \
        np.timedelta64(1, 'D')

    for i in range(4):
        df_sif['Latitude_corner_' + str(i + 1)] = \
            nc_fid.variables['Latitude_corners'][:][:, i]
        df_sif['Longitude_corner_' + str(i + 1)] = \
            nc_fid.variables['Longitude_corners'][:][:, i]

    if lat is None and lon is None:
        # @TODO
        pass

    return df_sif
