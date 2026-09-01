from haversine import haversine_vector
from tqdm import tqdm
import pandas as pd
import numpy as np



def init_track_dataframe(track_df):
    columns = ['ISO_TIME', 'LAT', 'LON', 'WS']
    if 'PROB' in track_df.columns: columns.append('PROB')
    # select only predicted cyclones
    track_df = track_df[(~np.isnan(track_df['LAT'])) & (~np.isnan(track_df['LON']))][columns].reset_index(drop=True)
    # add empty track id data
    track_df['TRACK_ID'] = ['' for _ in range(len(track_df))]
    # add empty distance with previous TC
    track_df['HAVERSINE'] = [np.inf for _ in range(len(track_df))]
    return track_df


def tracking_algorithm(detected_tracks, max_distance=400.0, min_track_count=12):
    # get all the iso times within the detected tracks
    iso_times = sorted(detected_tracks['ISO_TIME'].unique())
    for t in tqdm(range(len(iso_times)-1)):
        # init track progressive
        k = 0
        # get current detections
        cur_detections = detected_tracks[detected_tracks['ISO_TIME']==iso_times[t]]
        # get next detections
        next_detections = detected_tracks[detected_tracks['ISO_TIME']==iso_times[t+1]]

        # check of 6-hour timelag
        time_delta_days = (pd.to_datetime(iso_times[t+1]) - pd.to_datetime(iso_times[t])).days
        time_delta_hr = ((pd.to_datetime(iso_times[t+1]) - pd.to_datetime(iso_times[t])).seconds / 3600)

        # prepare track ids
        for i,detection in cur_detections.iterrows():
            # prepare a track id
            new_track_id = str(iso_times[t]).replace(' ', '-').replace('-','').split(':')[0]+f'_{k}'
            k += 1
            # if the detection is new, it will have a new track id
            if detection['TRACK_ID'] == '':
                # assign a new track id to this track
                detected_tracks.iloc[i, detected_tracks.columns.get_loc('TRACK_ID')] = new_track_id
                cur_detections.loc[i, 'TRACK_ID'] = new_track_id

        # check if the timelag is 6 hours
        if not time_delta_hr == 6.0 or time_delta_days != 0:
            continue

        # merge current and next detections 
        merge = pd.merge(cur_detections, next_detections, how='cross')
        # compute haversine distance
        merge['HAVERSINE'] = haversine_vector(array1=merge[['LAT_x','LON_x']].to_numpy().astype(np.float32), array2=merge[['LAT_y','LON_y']].to_numpy(), normalize=True)
        # sort the values by haversine distance
        merge = merge.sort_values(by='HAVERSINE').reset_index(drop=True)
        # remove too distant matches
        merge = merge[merge['HAVERSINE'] <= max_distance]

        while True:
            # skip if there are no correspondences left
            if len(merge) == 0: break
            for i,row in merge.iterrows():
                # assign the track id into the original dataframe
                if row['TRACK_ID_y'] == '':
                    detected_tracks.iloc[detected_tracks[(detected_tracks['LAT']==row['LAT_y']) & (detected_tracks['LON']==row['LON_y']) & (detected_tracks['ISO_TIME']==row['ISO_TIME_y'])].index, detected_tracks.columns.get_loc('TRACK_ID')] = row['TRACK_ID_x']
                    detected_tracks.iloc[detected_tracks[(detected_tracks['LAT']==row['LAT_x']) & (detected_tracks['LON']==row['LON_x']) & (detected_tracks['ISO_TIME']==row['ISO_TIME_x'])].index, detected_tracks.columns.get_loc('HAVERSINE')] = row['HAVERSINE']
                else:
                    continue
                # remove from merge the other y rows
                merge = merge[~((merge['ISO_TIME_x']==row['ISO_TIME_x']) & (merge['LAT_x']==row['LAT_x']) & (merge['LON_x']==row['LON_x']))]
                merge = merge[~((merge['ISO_TIME_y']==row['ISO_TIME_y']) & (merge['LAT_y']==row['LAT_y']) & (merge['LON_y']==row['LON_y']))]
                break

    # prepare track ids for last day
    cur_detections = detected_tracks[detected_tracks['ISO_TIME']==iso_times[-1]]
    k = 0
    for i,detection in cur_detections.iterrows():
        # prepare a track id
        new_track_id = str(iso_times[-1]).replace(' ', '-').replace('-','').split(':')[0]+f'_{k}'
        k += 1
        # if the detection is new, it will have a new track id
        if detection['TRACK_ID'] == '':
            # assign a new track id to this track
            detected_tracks.iloc[i, detected_tracks.columns.get_loc('TRACK_ID')] = new_track_id
            cur_detections.loc[i, 'TRACK_ID'] = new_track_id

    # remove too short tracks
    detected_tracks = detected_tracks.groupby('TRACK_ID').filter(lambda x: len(x) >= min_track_count).reset_index(drop=True)
    return detected_tracks
