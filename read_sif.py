"""
Read satellite data of solar induced fluorescence

(c) 2017 Wu Sun <wu.sun@ucla.edu>

- `read_gome2_l2`: Read GOME-2 SIF Level 2 data
- `read_gome2_l3`: Read GOME-2 SIF Level 3 data
- `read_oco2_l2`: Read OCO-2 Level 2 LiteSIF data

"""
import os
import numpy as np
import pandas as pd
import netCDF4


def great_arc(p1, p2, formula=None, radian=False):
    """
    Calculate the great arc length between two points on a spherical surface.

    """
    # force to be numpy array (for list or tuple input)
    p1 = np.array(p1, dtype='float64')
    p2 = np.array(p2, dtype='float64')
    # force the values to be in radian
    # note that `numpy.deg2rad()` function is faster than pure python!
    if not radian:
        p1 = np.deg2rad(p1)
        p2 = np.deg2rad(p2)

    # use transposed array to unpack column vectors
    # transposition has no influence on 1D array
    lat1, lon1 = p1.T
    lat2, lon2 = p2.T

    R_earth = 6.371e6  # mean radius, not equator radius, in m

    if formula is None:
        delta_sigma = np.arccos(
            np.sin(lat1) * np.sin(lat2) +
            np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2))
    elif formula is 'haversine':
        pass
    elif formula is 'vincenty':
        pass

    return R_earth * delta_sigma


def read_gome2_l2(filepath, lat=None, lon=None, dist_tolerance=50e3):
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
    # from filename, parse the date string and the satellite name
    filename = os.path.basename(filepath)

    if 'MOB' in filename:
        satellite = 'MetOp-B'
        date_str = filename.split('nolog.MOB.')[1][0:8]
    else:
        satellite = 'MetOp-A'
        date_str = filename.split('nolog.')[1][0:8]

    date_str = '-'.join([date_str[0:4], date_str[4:6], date_str[6:8]])

    # read the data and create `pandas.DataFrame` for reformatting
    nc_fid = netCDF4.Dataset(filepath, 'r')

    variable_names = [key for key in nc_fid.variables if key not in
                      ['time', 'Calibration_factor',
                       'Latitude_corners', 'Longitude_corners']]
    df_sif = pd.DataFrame(columns=variable_names)
    df_sif.insert(0, 'datetime', np.datetime64('nat'))

    datetime_start = np.datetime64(
        date_str + 'T' + ''.join(nc_fid.variables['time'][0].astype('|U1')))

    # do the point extraction when `lat` and `lon` parameters are both given
    if lat is not None and lon is not None:
        great_arc_dist = great_arc(
            [lat, lon],
            np.vstack([nc_fid.variables['latitude'][:],
                       nc_fid.variables['longitude'][:]]).T)
        min_dist = np.nanmin(great_arc_dist)
        if (min_dist > dist_tolerance) or np.isnan(min_dist):
            # @TODO: add warning information
            # will return a blank dataframe
            # still need to append the blank columns to conform to the format
            for i in range(4):
                df_sif['latitude_corner_' + str(i + 1)] = []
                df_sif['longitude_corner_' + str(i + 1)] = []
        else:
            nearest_point_index = np.nanargmin(great_arc_dist)
            time_str = ''.join(
                nc_fid.variables['time'][nearest_point_index].astype('|U1'))
            df_sif = df_sif.set_value(
                0, 'datetime', np.datetime64('T'.join([date_str, time_str])))
            if df_sif.loc[0, 'datetime'] < datetime_start:
                df_sif.loc[0, 'datetime'] += np.timedelta64(1, 'D')
            for var in variable_names:
                # must give a list or array in assigning columns
                df_sif = df_sif.set_value(
                    0, var, nc_fid.variables[var][nearest_point_index])
            # unpack lat/lon corners (not sure what those variables mean)
            for i in range(4):
                df_sif['latitude_corner_' + str(i + 1)] = nc_fid.variables[
                    'Latitude_corners'][:][nearest_point_index, i]
                df_sif['longitude_corner_' + str(i + 1)] = nc_fid.variables[
                    'Longitude_corners'][:][nearest_point_index, i]
    else:
        # won't do the point extraction and will return the whole dataset
        for var in variable_names:
            df_sif[var] = nc_fid.variables[var][:]
        # unpack lat/lon corners (not sure what those variables mean)
        for i in range(4):
            df_sif['latitude_corner_' + str(i + 1)] = \
                nc_fid.variables['Latitude_corners'][:][:, i]
            df_sif['longitude_corner_' + str(i + 1)] = \
                nc_fid.variables['Longitude_corners'][:][:, i]
        # convert char timestamps to `numpy.datetime64` format
        for i in range(df_sif.shape[0]):
            time_str = ''.join(nc_fid.variables['time'][i].astype('|U1'))
            df_sif = df_sif.set_value(
                i, 'datetime', np.datetime64('T'.join([date_str, time_str])))
        df_sif.loc[df_sif['datetime'] < datetime_start, 'datetime'] += \
            np.timedelta64(1, 'D')

    # store calibration factor and other meta information in `_metadata`
    # (requires pandas 0.13+)
    df_sif._metadata = {
        'calibration factor': nc_fid.variables['Calibration_factor'][:][0],
        'filename': filename,
        'satellite': satellite,
        'level': 2}

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

    panel_sif.rename(items={
        'cos(SZA)': 'cos_SZA',
        'Par_normalized_SIF_740': 'PAR_normalized_SIF_740',
        'Par_normalized_SIF_740_std': 'PAR_normalized_SIF_740_std',
        'Counts': 'counts'}, inplace=True)

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
