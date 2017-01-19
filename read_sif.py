"""
Read satellite data of solar induced fluorescence

(c) 2017 Wu Sun <wu.sun@ucla.edu>

- `read_gome2_l2`: Read GOME-2 SIF Level 2 data
- `read_gome2_l3`: Read GOME-2 SIF Level 3 data

"""
import os
import numpy as np
import pandas as pd
import netCDF4


def read_gome2_l2(filepath, lat=None, lon=None):
    """
    Read GOME-2 SIF Level 2 data (whole file or single point).

    Parameters
    ----------
    @TODO: to be completed

    Return
    ------
    df_sif : pandas.DataFrame


    Raise
    -----
    RuntimeError
        If the file is not found.

    """
    nc_fid = netCDF4.Dataset(filepath, 'r')

    variable_names = [key for key in nc_fid.variables if key not in
                      ['time', 'Calibration_factor',
                       'Latitude_corners', 'Longitude_corners']]
    df_sif = pd.DataFrame(columns=variable_names)

    for var in variable_names:
        df_sif[var] = nc_fid.variables[var][:]

    df_sif.insert(0, 'datetime', np.datetime64('nat'))

    filename = os.path.basename(filepath)
    if 'MOB' in filename:
        satellite = 'MetOp-B'
        date_str = filename.split('nolog.MOB.')[1][0:8]
    else:
        satellite = 'MetOp-A'
        date_str = filename.split('nolog.')[1][0:8]
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

    # store calibration factor and other meta information in `_metadata`
    # (requires pandas 0.13+)
    df_sif._metadata = {
        'calibration factor': nc_fid.variables['Calibration_factor'][:][0],
        'filename': filename,
        'satellite': satellite,
        'level': 2}

    if lat is None and lon is None:
        # @TODO: add the point extraction functionality
        pass

    return df_sif


def read_gome2_l3(filepath, lat=None, lon=None):
    """
    Read GOME-2 SIF Level 2 data (whole file or single point).

    Parameters
    ----------
    @TODO: to be completed

    Return
    ------
    panel_sif : pandas.Panel
        A 3D panel structure, the first dimension is the variable name
        (also known as 'item'), the second one is the latitude index, and
        the third one is the longitude.
        Note that 4D and N-dimensional panels have been deprecated. For
        concatenating multiple panel data objects to form a time series,
        please use the `xarray` package.

    Raise
    -----
    RuntimeError
        If the file is not found.

    """
    nc_fid = netCDF4.Dataset(filepath, 'r')

    variable_names = [key for key in nc_fid.variables
                      if key not in ['latitude', 'longitude']]
    n_lat, n_lon = nc_fid.variables['SIF_740'][:].shape
    latitude = nc_fid.variables['latitude'][:]
    longitude = nc_fid.variables['longitude'][:]

    panel_sif = pd.Panel(items=variable_names, major_axis=np.arange(n_lat),
                         minor_axis=np.arange(n_lon))

    for var in variable_names:
        panel_sif[var] = nc_fid.variables[var][:]

    panel_sif.rename(items={'cos(SZA)': 'cos_SZA'}, inplace=True)

    # cast latitude and longitude as 2D arrays
    panel_sif['latitude'] = np.repeat(np.array([latitude]).T, n_lon, axis=1)
    panel_sif['longitude'] = np.repeat(np.array([longitude]), n_lat, axis=0)

    filename = os.path.basename(filepath)
    if 'MOB' in filename:
        satellite = 'MetOp-B'
    else:
        satellite = 'MetOp-A'

    date_str = filename.split('_')[-2][0:8]
    date_str = '-'.join([date_str[0:4], date_str[4:6], date_str[6:8]])

    # store some meta data information
    panel_sif._metadata = {'date': date_str, 'filename': filename,
                           'satellite': satellite, 'level': 3}

    if lat is None and lon is None:
        # @TODO: add the point extraction functionality
        pass

    return panel_sif
