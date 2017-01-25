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


def great_arc(p1, p2, method=None, radian=False):
    """
    Calculate the great arc length between two points on a spherical surface.

    @TODO: add other methods
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

    if method is None:
        delta_sigma = np.arccos(
            np.sin(lat1) * np.sin(lat2) +
            np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2))
    elif method is 'haversine':
        pass
    elif method is 'vincenty':
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
    if lat is None or lon is None:
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
        # use element-wise concatenation rule on `numpy.chararray`
        time_chararr = nc_fid.variables['time'][:]
        datetime_chararr = np.chararray(time_chararr.shape[0], itemsize=11)
        datetime_chararr[:] = date_str + 'T'
        for i in range(time_chararr.shape[1]):
            datetime_chararr = datetime_chararr + time_chararr[:, i]
        # cast as `numpy.ndarray` and convert to `numpy.datetime64` type
        datetime_chararr = np.array(datetime_chararr).astype('|U')
        df_sif['datetime'] = datetime_chararr.astype(np.datetime64)
        df_sif.loc[df_sif['datetime'] < datetime_start, 'datetime'] += \
            np.timedelta64(1, 'D')
    else:
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

    # rectify differences in column names
    if 'SIF_740' in df_sif.columns and 'SIF_error' in df_sif.columns:
        df_sif.rename(columns={'SIF_error': 'SIF_740_error',
                               'SIF_uncorrected': 'SIF_740_uncorrected'},
                      inplace=True)
    if 'SIF_685' in df_sif.columns and 'SIF_error' in df_sif.columns:
        df_sif.rename(columns={'SIF_error': 'SIF_685_error',
                               'SIF_uncorrected': 'SIF_685_uncorrected'},
                      inplace=True)

    # store calibration factor and other meta information in `_metadata`
    # (requires pandas 0.13+)
    df_sif._metadata = {
        'calibration factor': nc_fid.variables['Calibration_factor'][:][0],
        'filename': filename,
        'satellite': satellite,
        'level': 2}

    return df_sif


def read_gome2_l3(filepath, lat=None, lon=None, dist_tolerance=56e3):
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

    df_sif : pandas.DataFrame
        If point extraction

    Raise
    -----
    RuntimeError
        If the file is not found.

    """
    # from filename, parse the date string and the satellite name
    filename = os.path.basename(filepath)
    if 'MOB' in filename:
        satellite = 'MetOp-B'
    else:
        satellite = 'MetOp-A'

    date_str = filename.split('_')[-2][0:8]
    date_str = '-'.join([date_str[0:4], date_str[4:6], date_str[6:8]])

    # read the data and create `pandas.DataFrame` for reformatting
    nc_fid = netCDF4.Dataset(filepath, 'r')

    variable_names = [key for key in nc_fid.variables
                      if key not in ['latitude', 'longitude']]
    n_lat, n_lon = nc_fid.variables['SIF_740'][:].shape
    latitude = nc_fid.variables['latitude'][:]
    longitude = nc_fid.variables['longitude'][:]

    # cast latitude and longitude as 2D arrays
    grid_lat = np.repeat(np.array([latitude]).T, n_lon, axis=1)
    grid_lon = np.repeat(np.array([longitude]), n_lat, axis=0)

    if lat is None or lon is None:
        # won't do the point extraction and will return the whole dataset
        panel_sif = pd.Panel(items=variable_names, major_axis=np.arange(n_lat),
                             minor_axis=np.arange(n_lon))

        for var in variable_names:
            panel_sif[var] = nc_fid.variables[var][:]

        panel_sif.rename(items={
            'cos(SZA)': 'cos_SZA',
            'Par_normalized_SIF_740': 'PAR_normalized_SIF_740',
            'Par_normalized_SIF_740_std': 'PAR_normalized_SIF_740_std',
            'Counts': 'counts'}, inplace=True)

        panel_sif['latitude'] = grid_lat
        panel_sif['longitude'] = grid_lon

        # 'counts' should be integers
        panel_sif['counts'] = panel_sif['counts'].astype(np.int64)
        # use proper NaN notation for missing data
        panel_sif.replace(-999., np.nan, inplace=True)
        # store some meta data information
        panel_sif._metadata = {'date': date_str, 'filename': filename,
                               'satellite': satellite, 'level': 3}

        return panel_sif
    else:
        df_sif = pd.DataFrame(columns=variable_names)
        great_arc_dist = great_arc(
            [lat, lon], np.vstack([grid_lat.flatten(), grid_lon.flatten()]).T)
        min_dist = np.nanmin(great_arc_dist)

        if (min_dist > dist_tolerance) or np.isnan(min_dist):
            # @TODO: add warning information
            # will return a blank dataframe
            df_sif['latitude'] = []
            df_sif['longitude'] = []
        else:
            nearest_point_index = np.nanargmin(great_arc_dist)
            nearest_point_indices = (nearest_point_index // n_lon,
                                     nearest_point_index % n_lon)
            df_sif = df_sif.set_value(
                0, 'latitude', grid_lat[nearest_point_indices])
            df_sif = df_sif.set_value(
                0, 'longitude', grid_lon[nearest_point_indices])

            for var in variable_names:
                df_sif = df_sif.set_value(
                    0, var, nc_fid.variables[var][:][nearest_point_indices])

            df_sif.rename(columns={
                'cos(SZA)': 'cos_SZA',
                'Par_normalized_SIF_740': 'PAR_normalized_SIF_740',
                'Par_normalized_SIF_740_std': 'PAR_normalized_SIF_740_std',
                'Counts': 'counts'}, inplace=True)

        # 'counts' should be integers
        df_sif['counts'] = df_sif['counts'].astype(np.int64)
        # use proper NaN notation for missing data
        df_sif.replace(-999., np.nan, inplace=True)
        # store some meta data information
        df_sif._metadata = {'date': date_str, 'filename': filename,
                            'satellite': satellite, 'level': 3}

        return df_sif
