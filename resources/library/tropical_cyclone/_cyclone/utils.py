# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The GPLv3 License [see LICENSE for details]#
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from scipy.ndimage import gaussian_filter
from global_land_mask import globe
import xarray as xr
import pandas as pd
import numpy as np

from tropical_cyclone._cyclone.macros import DENSITY_MAP_TC, SQUARE_MAP_TC, LABEL_MAP_TC
from tropical_cyclone._cyclone.georef import get_global_rowcol_coords, from_local_to_global
from tropical_cyclone.utils import round_to_grid


def get_tropical_cyclone_positions(ds:xr.Dataset, georef:pd.DataFrame, label:pd.DataFrame=None, square:bool=False, sigma:int=10, radius:int=10, coords=('latitude','longitude')) -> xr.Dataset:
    """
    Add Tropical Cyclone maps into the passed xr.Dataset. The maps built from `georef` and `label`
    pd.DataFrames are 3:
    1. `dm_tc` : map that, for every TC position has a gaussian filter of dimension 
        `sigma` pixels around its center.
    2. `sq_tc` - map that, for every TC position has a square of `radius` pixels around
        its center.
    3. `lb_tc` - map that, for every TC position has a square of `radius` pixels around
        its center, whose value is the same across each cyclone SID. This means that the 
        label correspondence between each TC occurrence is preserved among different timesteps.

    Parameters
    ----------
    ds : xr.Dataset
        Xarray dataset in which will be added the maps
    georef : pd.DataFrame
        Georeferencing dataset that contains the positions of each TC

    Returns
    -------
    ds : xr.Dataset
        The updated dataset

    Examples
    --------
    >>> ds = xr.open_dataset('path/to/netcdf/dataset')
    >>> georef = pd.DataFrame(data={'ISO_TIME':[], 'SID':[], 'LAT':[], 'LON':[], 'RLAT':[], 'RLON':[], 'YLAT':[], 'XLON':[]})
    >>> label = tc.cyclone.get_tc_map_labels(georef)
    >>> ds = tc.cyclone.get_tropical_cyclone_positions(ds, georef, label)

    """
    lat, lon = coords
    # make a copy of georeferencing pd.DataFrame
    gr = georef.copy()
    # convert gr column into datetime
    gr['ISO_TIME'] = pd.to_datetime(gr['ISO_TIME'])
    # get the shape of the 2D cyclone maps to be added to the xr.Dataset
    shape = (ds['time'].shape[0], ds[lat].shape[0], ds[lon].shape[0])
    # create a gaussian map that will contain the density map of each TC
    gaussian_map = np.zeros(shape=shape)
    # create a block map that will contain a square centered on each TC
    if square: square_map = np.zeros(shape=shape)
    # create a label map that will contain correspondences between TCs among timesteps
    if label is not None: label_map = np.zeros(shape=shape)
    # select only iso times that are compatible with our dataset
    gr = gr[gr['ISO_TIME'].isin(pd.to_datetime(ds['time']))]
    # loop over each time-step
    for t,iso_time in enumerate(gr['ISO_TIME'].unique()):
        rows = gr[gr['ISO_TIME']==iso_time]
        for _,row in rows.iterrows():
            j,i = row['XLON'], row['YLAT']
            # place 1 on the TC center
            gaussian_map[t,i,j] = 1
            # place 1 over a square centered on the TC center
            if square: square_map[t, i-radius:i+radius+1, j-radius:j+radius+1] = 1
            # place label value over a square on the TC center
            # retrieve the column in which we find the selected SID
            if label is not None: label_map[t, i-radius:i+radius+1, j-radius:j+radius+1] = int(label.eq(row['SID']).any().argmax())
        # apply gaussian filtering over timestep t
        gaussian_map[t,:,:] = gaussian_filter(gaussian_map[t,:,:], sigma=sigma)
        # rescale the density map in [0,1]
        gaussian_map[t,:,:] = (gaussian_map[t,:,:] - gaussian_map[t,:,:].min()) / (gaussian_map[t,:,:].max() - gaussian_map[t,:,:].min())
    # create a density map in the xr.Dataset
    ds[DENSITY_MAP_TC] = (('time',lat,lon), gaussian_map)
    # create a square map in the xr.Dataset
    if square: ds[SQUARE_MAP_TC] = (('time',lat,lon), square_map)
    # create a label map in the xr.Dataset
    if label is not None: ds[LABEL_MAP_TC] = (('time',lat,lon), label_map)
    # return the updated dataset
    return ds


def retrieve_predicted_tc(
        y_pred,                # TC locations
        ds,                    # the original dataset with `lat` and `lon`
        patch_ds,              # the patch dataset to store the TC centers
        patch_size,            # the size of a patch
        eps: float = 0.1,      # the confidence for negative values
        y_pred_cls = None,     # if provided, it is the probability for having a TC in the patch
        y_pred_loc_std = None, # if provided, it is the std associated to the TC locations
        ):
    """
    Retrieves the latitude-longitude coordinates from the passed predicted TCs

    eps: float = 0.1 - small value > 0

    """
    # create a latlons matrix with the same shape of y_pred_reshaped filled with nan
    cyclone_latlon_coords = np.full_like(y_pred, fill_value=np.nan)
    cyclone_rowcol_coords = np.full_like(y_pred, fill_value=np.nan)
    # for each timestep
    for t in range(y_pred.shape[0]):
        # for each row
        for i in range(y_pred.shape[1]):
            # for each column
            for j in range(y_pred.shape[2]):
                # correct the prediction if `eps` <= x <= 0.0 (slightly negative, could be an oscillation)
                if y_pred[t,i,j,0] < 0.0:
                    if y_pred[t,i,j,0] >= -eps:
                        y_pred[t,i,j,0] = 0.0
                if y_pred[t,i,j,1] < 0.0:
                    if y_pred[t,i,j,1] >= -eps:
                        y_pred[t,i,j,1] = 0.0
                # correct the prediction if x > 39.0 (too high, could be an oscillation)
                if y_pred[t,i,j,0] >= patch_size - 1:
                    y_pred[t,i,j,0] = patch_size - 1
                if y_pred[t,i,j,1] >= patch_size - 1:
                    y_pred[t,i,j,1] = patch_size - 1
                # if the model prediction is valid
                if y_pred[t,i,j,0] >= 0.0 and y_pred[t,i,j,1] >= 0.0:
                    try:
                        # retrieve global row-col coordinates of the TC
                        global_rowcol = from_local_to_global((i,j), y_pred[t,i,j,:], patch_size)
                        # retrieve global lat-lon coordinates of the TC
                        cyclone_latlon_coords[t,i,j,:] = (ds['lat'].data[global_rowcol[0]], ds['lon'].data[global_rowcol[1]])
                        cyclone_rowcol_coords[t,i,j,:] = (global_rowcol[0], global_rowcol[1])
                    except:
                        continue
    # update patch_ds cyclone_information
    patch_ds['patch_cyclone_pred'] = (('time','rows','cols','coordinate'), cyclone_latlon_coords)
    patch_ds['patch_rowcol_pred'] = (('time','rows','cols','rowcol'), cyclone_rowcol_coords)
    if y_pred_cls is not None:
        patch_ds['patch_cyclone_probability'] = (('time','rows','cols','prob'), y_pred_cls)
    if y_pred_loc_std is not None:
        patch_ds['patch_cyclone_pred_std'] = (('time','rows','cols','coordinate'), y_pred_loc_std)
    return patch_ds


def filter_basin(tracks):
    tracks['basin'] = ''
    first_tracks = tracks.groupby(by='track_id').first().reset_index()
    first_tracks.loc[first_tracks['lon'] <= 180.0, 'basin'] = 'WNP'
    first_tracks.loc[first_tracks['lon'] > 180.0, 'basin'] = 'ENP'
    tracks.loc[tracks['track_id'].isin(list(first_tracks[first_tracks['basin']=='WNP']['track_id'].to_numpy())), 'basin'] = 'WNP'
    tracks.loc[tracks['track_id'].isin(list(first_tracks[first_tracks['basin']=='ENP']['track_id'].to_numpy())), 'basin'] = 'ENP'
    return tracks


def filter_tracks(tracks, dates, lat_range, lon_range, min_track_count, remove_atlantic: bool = False, remove_short_tracks: bool = False):
    # get only observations in dates
    tracks = tracks[tracks['time'].isin(dates)].reset_index(drop=True)
    # get only tracks within the lat-lon ranges
    tracks = tracks[(tracks['lon']>=lon_range[0]) & (tracks['lon']<=lon_range[1]) & (tracks['lat']>=lat_range[0]) & (tracks['lat']<=lat_range[1])]
    # remove atlantic basin
    if remove_atlantic:
        tracks = tracks[~((tracks['lon'] >= 260) & (tracks['lat'] >= 20))]
        tracks = tracks[tracks['lon'] <= 270]
    # convert longitudes to range [0, 360] format
    tracks['lon'] = (tracks['lon'] + 180) % 360 - 180
    # remove too short tracks
    if remove_short_tracks:
        tracks = tracks[tracks['track_id'].isin(tracks.groupby('track_id').filter(lambda x: len(x) >= min_track_count)['track_id'].unique())].reset_index(drop=True)
    return tracks


def compute_pod_and_far(dynamicopy, algo, algo_id, obs, max_dist, print_results=False):
    matches = dynamicopy.match_tracks(algo, obs, algo_id, 'ibtracs', max_dist=max_dist, min_overlap=0, ref=True)
    
    n_gt_match = len(matches[f'id_ibtracs'].unique())
    n_algo_match = len(matches[f'id_{algo_id}'].unique())
    n_observations = len(obs.track_id.unique())
    n_detections = len(algo.track_id.unique())
    
    H, M, FA = n_gt_match, n_observations - n_gt_match, n_detections - n_algo_match
    POD = H / (H + M)
    FAR = FA / (H + FA)
    
    matches = matches.rename(columns={'temp':'matching_timesteps'})
    
    if print_results:
        print(f'Algorithm : {algo_id.upper()}\n   Hits : {H}\n   Misses : {M}\n   False Alarms : {FA}\n   POD : {POD}\n   FAR : {FAR}\n')
    return matches, pd.DataFrame(data={
            'algo':[algo_id.upper()], 
            'hits':[H], 
            'misses':[M], 
            'false alarms':[FA], 
            'pod':[POD], 
            'far':[FAR]})


def compute_storm_transits(tracks, bins, lats, lons, norm_month=False):
    # make a copy of the dataframe
    tracks_0_360 = tracks.copy()
    # convert longitudes from [-180,180] to [0, 360]
    tracks_0_360['lon'] = ((tracks_0_360['lon'] + 360) % 360).to_numpy()
    # get rowcol coordinates of the cyclone centers
    rowcols = get_global_rowcol_coords(lats=lats, lons=lons, latlon=round_to_grid(tracks_0_360[['lat','lon']].to_numpy()))
    # compute storm transits  for each bin with the histogram
    storm_transits, _, _ = np.histogram2d(rowcols[:,0], rowcols[:,1], bins=bins)
    # normalize with respect to the number of months
    if norm_month: storm_transits = storm_transits // 12
    # remove points where we do not have a transit
    storm_transits = np.where(storm_transits==0, np.nan, storm_transits)
    return storm_transits


def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def remove_tcs_above_x_lat(df: pd.DataFrame, lat: float):
    to_remove_track_ids = []
    for track_id in df[df['lat'] >= lat]['track_id'].unique():
        if df[df['track_id'] == track_id].sort_values(by='time').iloc[0]['lat'] >= lat:
            to_remove_track_ids.append(track_id)
    df = df[~df['track_id'].isin(to_remove_track_ids)].reset_index(drop=True)
    return df


def remove_tcs_originated_on_land(df: pd.DataFrame):
    to_remove_track_ids = []
    for track_id in df['track_id'].unique():
        first_tc_occurrence = df[df['track_id']==track_id].sort_values(by='time').iloc[0]
        lat, lon = first_tc_occurrence['lat'], first_tc_occurrence['lon']
        if globe.is_land(lat, lon):
            to_remove_track_ids.append(track_id)
    df = df[~df['track_id'].isin(to_remove_track_ids)].reset_index(drop=True)
    return df
