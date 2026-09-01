# --------------------------------------------------------
# Machine Learning Library
# Copyright (c) 2021 *********
# Licensed under The *** License [see LICENSE for details]
# Written by Davide Donno
# --------------------------------------------------------
from math import atan2, radians, degrees, sin, cos
from scipy.ndimage import gaussian_filter
from global_land_mask import globe
import xarray as xr
import pandas as pd
import numpy as np
import random
import re
import os

from tropical_cyclone._cyclone.macros import DENSITY_MAP_TC, SQUARE_MAP_TC, LABEL_MAP_TC
from tropical_cyclone._cyclone.georef import from_local_to_global, get_global_rowcol_coords
from tropical_cyclone.utils import round_to_grid


mean = {
    'fg10': 9.29946005e+00, 
    'i10fg': 8.63399272e+00, 
    'msl': 1.01096454e+05, 
    't_500': 2.65829426e+02,
    't_300': 2.40452589e+02, 
    'vo_850': 2.56295123e-06, 
}
std = {
    'fg10': 4.49108529e+00, 
    'i10fg': 4.40210274e+00, 
    'msl': 6.0234259e+02, 
    't_500': 5.8706489e+00, 
    't_300': 5.5995750e+00, 
    'vo_850': 4.86139457e-05, 
}


def compute_euclidean_distance(TPs_true, TPs_pred, avg=True, grid_resolution=0.25, km_to_deg=110.474):
    # compute euclidean distance on TPs and convert it in degrees and kilometers
    euc_dist_on_tps_px = np.linalg.norm(TPs_true - TPs_pred)
    euc_dist_on_tps_deg = euc_dist_on_tps_px * grid_resolution
    euc_dist_on_tps_km = euc_dist_on_tps_deg * km_to_deg

    if avg:
        # reduce mean to get average values
        euc_dist_on_tps_px = np.mean(euc_dist_on_tps_px)
        euc_dist_on_tps_deg = np.mean(euc_dist_on_tps_deg)
        euc_dist_on_tps_km = np.mean(euc_dist_on_tps_km)

    return euc_dist_on_tps_px, euc_dist_on_tps_deg, euc_dist_on_tps_km


def get_ibtracs_by_basins(ibtracs_df, basins=['WP', 'EP', 'NA']):
    """
    Given the list of basins, return the IBTrACS DataFrame

    Parameters
    ----------

    ibtracs_df : pd.DataFrame
        ibtracs DataFrame containing all basins information

    basins : list(str)
        List of basins from which we want to extract basin information.
    
    """
    # get SIDs list for each basin
    SIDs_list = [ibtracs_df[ibtracs_df['BASIN']==basin]['SID'].unique().tolist() for basin in basins]
    # convert SIDs list into sets
    SIDs_sets = [set(SIDs) for SIDs in SIDs_list]
    
    # compute difference between basins to find out basin-only TCs
    SIDs_basin_only = {}
    for sids, basin in zip(SIDs_sets, basins):
        # get list of SIDs present only in that basin
        for s,b in zip(SIDs_sets,basins):
            # skip if b is equal to basin
            if basin == b:
                continue
            # remove from sids the other basins
            sids = sids.difference(s)
        SIDs_basin_only.update({basin:sids})

    # compute intersection between basins to find out cross-basin TCs
    SIDs_basin_intersection = {}
    if len(basins) > 1:
        for sids, basin in zip(SIDs_sets, basins):
            # get list of SIDs present only in that basin
            for s,b in zip(SIDs_sets,basins):
                # skip if b is equal to basin
                if basin == b:
                    continue
                if not basin+'_and_'+b in list(SIDs_basin_intersection.keys()) and not b+'_and_'+basin in list(SIDs_basin_intersection.keys()):
                    # intersection between sids and the other's basins sids
                    SIDs_basin_intersection.update({basin+'_and_'+b:sids.intersection(s)})
    
    # get only records containing the already extracted SIDs
    ibtracs_only_dfs = {}
    for basin,sids in SIDs_basin_only.items():
        ibtracs_only_dfs.update({basin:ibtracs_df[ibtracs_df['SID'].isin(sids)]})
    ibtracs_intersection_dfs = {}
    for basin,sids in SIDs_basin_intersection.items():
        ibtracs_intersection_dfs.update({basin:ibtracs_df[ibtracs_df['SID'].isin(sids)]})
    
    return ibtracs_only_dfs, ibtracs_intersection_dfs


def classification_score(y_pred, y_true, label_no_cyclone):
    acc = 0
    for i in range(y_pred.shape[0]):
        if(((y_pred[i,0] < 0) and (y_true[i,0]  == label_no_cyclone) and (y_pred[i,1] < 0) and (y_true[i,1]  == label_no_cyclone)) or ((y_pred[i,0] >= 0) and (y_true[i,0] >= 0) and (y_pred[i,1] >= 0) and (y_true[i,1] >= 0))):
            acc = acc+1
    acc = acc/y_pred.shape[0]
    return acc


def get_tc_map_labels(georef:pd.DataFrame, max_ts_tcs:int=20) -> pd.DataFrame:
    """
    Create a label pd.DataFrame in which, for each unique `ISO_TIME` in `georef`,
    it is associated the same label to each TC `SID` contained in georef.

    """
    # create a label pd.DataFrame
    label = pd.DataFrame(data={'ISO_TIME':georef['ISO_TIME'].unique()})
    for l in range(max_ts_tcs):
        # create a label to the label dataframe
        label[f'{l+1}'] = [np.nan] * len(label)
    # for each of the TCs in the dataset
    for sid in sorted(georef['SID'].unique()):
        # get the iso times in which it has developed
        iso_times = georef[georef['SID']==sid]['ISO_TIME'].sort_values().to_numpy()
        # get the first iso time in the label pd.DataFrame
        labels = label[label['ISO_TIME']==iso_times[0]].iloc[0,1:]
        # get the first nan index within the labels
        first_nan_idx = pd.isna(labels).argmax(axis=0)
        # add the TC SID to the corresponding label `first_nan_idx`
        label.loc[label['ISO_TIME'].isin(iso_times), f'{first_nan_idx+1}'] = sid
    return label


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


def build_tc_filenames(dataset_src, ibtracs_src, lead_time:int, years:list, extension:str='tf'):
    """
    Reading IBTrACS actually available iso times, builds a list of filenames for inputs and outputs

    """
    # compute time delta t
    dt = pd.DateOffset(hours=lead_time)
    # read ibtracs dataset
    ibtracs = pd.read_csv(ibtracs_src, index_col=0)
    ibtracs['ISO_TIME'] = pd.to_datetime(ibtracs['ISO_TIME'])
    # get all the unique iso times from ibtracs
    T = pd.Series(ibtracs['ISO_TIME'].unique())
    # remove from `T` those timesteps not in years
    T = T[T.dt.year.isin(years)]
    # lag the unique iso times with the delta t
    Tdt = np.array([np.datetime64(t + dt) for t in T])
    # convert T back to an array
    T = np.array(T)
    # create a pd.DataFrame holding the dates correspondences
    times = pd.DataFrame(data={'T' : list(T), 'T+dt' : list(Tdt)})
    # get only the lagged times that are actually present in IBTrACS
    available_times = times[times['T+dt'].isin(times['T'])].copy()
    # create a list of filenames at time t
    available_times['filenames(t)'] = [os.path.join(dataset_src, date.replace(' ','-').replace('-','_').split(':')[0]+f'.{extension}') for date in available_times['T'].astype(str)]
    # create a list of filenames at time t+dt
    available_times[f'filenames(t+dt)'] = [os.path.join(dataset_src, date.replace(' ','-').replace('-','_').split(':')[0]+f'.{extension}') for date in available_times['T+dt'].astype(str)]
    # reset the index of the pd.DataFrame
    available_times = available_times.reset_index(drop=True)
    # return lists of filenames(t) and filenames(t+dt)
    return list(available_times['filenames(t)'].to_numpy()), list(available_times['filenames(t+dt)'].to_numpy())


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
    # add new column named `basin`
    tracks['basin'] = ''
    # group tracks by track id and get the first element of the track
    first_tracks = tracks.groupby(by='track_id').first().reset_index()
    # determine the basin of belonging depending on the LON threshold
    first_tracks.loc[first_tracks['lon'] <= 180.0, 'basin'] = 'WNP'
    first_tracks.loc[first_tracks['lon'] > 180.0, 'basin'] = 'ENP'
    # assign the basin name to every track that according to the first track basin
    tracks.loc[tracks['track_id'].isin(list(first_tracks[first_tracks['basin']=='WNP']['track_id'].to_numpy())), 'basin'] = 'WNP'
    tracks.loc[tracks['track_id'].isin(list(first_tracks[first_tracks['basin']=='ENP']['track_id'].to_numpy())), 'basin'] = 'ENP'
    # return the tracks
    return tracks


def filter_tracks(
        tracks, 
        dates, 
        lat_range, 
        lon_range, 
        min_track_count, 
        remove_atlantic: bool = False, 
        remove_short_tracks: bool = False, 
    ):
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
    if H + M == 0:
        POD = 0.0
    else:
        POD = H / (H + M)
    FAR = FA / (H + FA)
    SR = 1 - FAR
    if POD == 0:
        CSI = 0.0
    else:
        CSI = ( 1 / ( ( 1 / POD ) + ( 1 / SR ) - 1 ) )
    if SR == 0:
        BIAS = np.inf
    else:
        BIAS = POD / SR
    matches = matches.rename(columns={'temp':'matching_timesteps'})
    
    if print_results:
        print(f'Algorithm : {algo_id.upper()}\n   Hits : {H}\n   Misses : {M}\n   False Alarms : {FA}\n   POD : {POD}\n   FAR : {FAR}\n')
    return matches, pd.DataFrame(data={
            'algo':[algo_id.upper()], 'hits':[H], 'misses':[M], 'false alarms':[FA], 
            'pod':[POD], 'far':[FAR], 'sr':[SR], 'csi':[CSI], 'bias':[BIAS]
        })


def compute_storm_transits(tracks, bins, range, lats, lons, norm_month=False):
    # make a copy of the dataframe
    tracks_0_360 = tracks.copy()
    # convert longitudes from [-180,180] to [0, 360]
    tracks_0_360['lon'] = ((tracks_0_360['lon'] + 360) % 360).to_numpy()
    # get rowcol coordinates of the cyclone centers
    rowcols = get_global_rowcol_coords(lats=lats, lons=lons, latlon=round_to_grid(tracks_0_360[['lat','lon']].to_numpy()))
    # compute storm transits  for each bin with the histogram
    storm_transits, _, _ = np.histogram2d(rowcols[:,0], rowcols[:,1], bins=bins, range=range)
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


def generate_random_origin(xtc, ytc, x_low, y_low, x_high, y_high, patch_size = 40, size = None):
    kwargs = {}
    if size: kwargs['size'] = size
    xi = np.random.randint(x_low, np.where(x_high - (patch_size - 1) < xtc, x_high - (patch_size - 1), xtc) + 1, **kwargs)
    yi = np.random.randint(y_low, np.where(y_high - (patch_size - 1) < ytc, y_high - (patch_size - 1), ytc) + 1, **kwargs)
    return xi, yi


def generate_random_coordinate(height: int, width: int, excluded_regions: tuple, patch_size: int, size: int):
    """     NOTE: GENERATO CON CHATGPT E MODIFICATO
    Genera una coppia (x, y) casuale fuori dalle regioni escluse.
    
    Parameters:
    - H, W: dimensioni della mappa
    - excluded_regions: lista di tuple (x_low, y_low, x_high, y_high)
    
    Returns:
    - (x, y): coordinate casuali valide
    """
    def gen_random_coo(height, width, excluded_regions, patch_size):
        while True:
            # generate random coordinate
            x, y = random.randint(0, width - patch_size - 1), random.randint(0, height - patch_size - 1)
            # flag to check if the coordinate is in 
            inside_excluded = False
            for (x_low, y_low, x_high, y_high) in excluded_regions:
                if x_low <= x <= x_high and y_low <= y <= y_high:
                    inside_excluded = True
                    break
            if not inside_excluded:
                return (x, y)
    coords = []
    for i in range(excluded_regions.shape[0]):
        x_low, y_low, x_high, y_high = excluded_regions[i]
        excluded_region = ((x_low - patch_size, y_low - patch_size, x_high + patch_size, y_high + patch_size), )
        ex_reg_coords = np.stack([gen_random_coo(height, width, excluded_region, patch_size) for _ in range(size)], axis=0)
        coords.append(ex_reg_coords)
    return np.stack(coords, axis=0)


def parse_bboxes_from_str_to_array(s):
    s_clean = re.sub(r'[\[\]]', '', s)  # rimuove le parentesi quadre
    return np.fromstring(s_clean, sep=' ')


def remove_tcs_above_x_lat(df: pd.DataFrame, lat: float):
    to_remove_track_ids = []
    if lat < 0.0:
        for track_id in df[df['lat'] <= lat]['track_id'].unique():
            if df[df['track_id'] == track_id].sort_values(by='time').iloc[0]['lat'] <= lat:
                to_remove_track_ids.append(track_id)
        df = df[~df['track_id'].isin(to_remove_track_ids)].reset_index(drop=True)
        return df
    else:
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


def remove_tcs_above_x_lat_gp(
        df: pd.DataFrame,
        gen_points: pd.DataFrame,
        lat: float,
    ):
    # Get all the TCs whose genesis point 
    # is above (below) the given lat
    if lat < 0.0:
        gen_points = gen_points[gen_points['lat'] <= lat]
    else:
        gen_points = gen_points[gen_points['lat'] >= lat]
    # Return the original dataframe with removed track ids
    return df[~df['track_id'].isin(gen_points['track_id'])]


def remove_tcs_originated_on_land_gp(
        df: pd.DataFrame,
        gen_points: pd.DataFrame,
    ):
    # Create a boolean mask to remove genesis points on land
    mask = globe.is_land(gen_points['lat'], gen_points['lon'])
    # Remove the mask values where we have True (is land)
    gen_points = gen_points[mask]
    return df[~df['track_id'].isin(gen_points['track_id'])]


def clean_tracks(
        dynamicopy, 
        tracks: pd.DataFrame,
        min_track_count: int = 12, 
    ):
    # Rename columns
    tracks = tracks.rename(columns={col: col.lower() for col in tracks.columns}).rename(
        columns={
            "iso_time": "time",
        }
    )

    # After BYTE algorithm we end up in having
    # some duplicated rows. Therefore, we remove
    # them and keep only unique rows. This
    # operation doesn't affect ByteStorm skills
    tracks = tracks.drop_duplicates()

    # Convert the time to pandas
    tracks['time'] = pd.to_datetime(tracks['time'])

    # Geographical details
    tracks.loc[tracks.lon < 0, "lon"] += 360
    tracks["hemisphere"] = np.where(tracks.lat > 0, "N", "S")
    tracks["basin"] = dynamicopy.get_basin(tracks.lon, tracks.lat)

    # All about time
    tracks["hour"] = tracks.time.dt.hour
    tracks["day"] = tracks.time.dt.day
    tracks["month"] = tracks.time.dt.month
    tracks["year"] = tracks.time.dt.year
    tracks = dynamicopy.add_season(tracks)

    # Get genesis points over the tracks
    gen_points = dynamicopy.genesis_points(tracks).reset_index()

    # Remove TCs originated above 30 LAT
    eps = 1.0 # allow error of eps degree
    tracks = remove_tcs_above_x_lat_gp(tracks, gen_points, 30+eps)

    # Get genesis points over the tracks
    gen_points = dynamicopy.genesis_points(tracks).reset_index()

    # Remove TCs originated above 30 LAT or below -30 LAT
    eps = 1.0 # allow error of eps degree
    tracks = remove_tcs_above_x_lat_gp(tracks, gen_points, -30-eps)

    # Convert longitudes to range [-180, 180] format
    tracks['lon'] = (tracks['lon'] + 180) % 360 - 180

    # Get (again) genesis points over the tracks
    gen_points = dynamicopy.genesis_points(tracks).reset_index()

    # Remove TCs originated on land
    tracks = remove_tcs_originated_on_land_gp(tracks, gen_points)

    # Geographical details
    tracks.loc[tracks.lon < 0, "lon"] += 360

    ## Filter tracks lasting for at least X time steps
    if min_track_count is not None:
        tcs = (
            tracks.groupby("track_id")["time"]
            .count()[tracks.groupby("track_id")["time"].count() >= min_track_count]
            .index
        )
        tracks = tracks[tracks.track_id.isin(tcs)]

    # Reset the index
    tracks = tracks.reset_index(drop=True)
    return tracks


def remove_short_tracks(
        tracks: pd.DataFrame, 
        min_track_count: int = 12, 
    ):
    tracks = tracks[tracks['track_id'].isin(tracks.groupby('track_id').filter(lambda x: len(x) >= min_track_count)['track_id'].unique())].reset_index(drop=True)
    return tracks

def convert_coords_to_format(tracks, format: str = '[-180,180]'):
    if format == '[-180,180]':
        tracks['lon'] = (tracks['lon'] + 180) % 360 - 180
    elif format == '[0,360]':
        tracks.loc[tracks.lon < 0, "lon"] += 360
    return tracks

def bearing(lat1, lon1, lat2, lon2):
    """Calcola il bearing (angolo in gradi) tra due coordinate geografiche"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon)
    brng = atan2(x, y)
    return (degrees(brng) + 360) % 360

def track_smoothness(latitudes, longitudes):
    bearings = []
    for i in range(1, len(latitudes)):
        brng = bearing(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
        bearings.append(brng)
    # Calcolo variazione dell'angolo tra step successivi
    delta_bearings = [abs(bearings[i] - bearings[i-1]) for i in range(1, len(bearings))]
    # Normalizza angoli a [0, 180]
    delta_bearings = [min(d, 360 - d) for d in delta_bearings]
    # Misura di "non smoothness": deviazione standard o media
    smoothness_score = np.std(delta_bearings)  # più piccolo = più smooth
    return smoothness_score, delta_bearings