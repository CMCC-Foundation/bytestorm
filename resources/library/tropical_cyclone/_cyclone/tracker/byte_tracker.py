# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 
# BYTE algorithm, taken and adapted from
# https://github.com/FoundationVision/ByteTrack/tree/main
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from tropical_cyclone._cyclone.tracker.basetrack import BaseTrack, TrackState
from tropical_cyclone._cyclone.tracker.kalman_filter import KalmanFilter
from tropical_cyclone._cyclone.tracker import matching

import pandas as pd
import numpy as np


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, lats, lons):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        
        # store latitude and longitudes
        self.lats = lats
        self.lons = lons
        # init the track dataframe
        self.track = pd.DataFrame(columns=['ISO_TIME', 'TRACK_ID', 'LAT', 'LON', 'BBOX'])

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, date=None):
        if date is None: 
            raise ValueError
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        # update the track dataframe to contain the information about the TC
        self.update_track_dataframe(date)

    def re_activate(self, new_track, frame_id, new_id=False, date=None):
        if date is None: 
            raise ValueError
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        # update the track dataframe to contain the information about the TC
        self.update_track_dataframe(date)

    def update(self, new_track, frame_id, date=None):
        if date is None: 
            raise ValueError
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        # update the track dataframe to contain the information about the TC
        self.update_track_dataframe(date)
        

    def update_track_dataframe(self, date):
        w1, h1, w2, h2 = self.tlbr
        w, h = (
            int(np.round(w1 + (w2 - w1) / 2)), 
            int(np.round(h1 + (h2 - h1) / 2))
            )
        if w >= self.lons.shape[0]: w = self.lons.shape[0] - 1
        if h >= self.lats.shape[0]: h = self.lats.shape[0] - 1
        lat, lon = self.lats[h], self.lons[w]
        # add the new track element to the dataframe
        self.track = pd.concat([self.track, pd.DataFrame(data={
            'ISO_TIME': [date], 
            'TRACK_ID': [str(self)], 
            'LAT': [lat], 
            'LON': [lon], 
            'BBOX': [self.tlbr], 
        })]).reset_index(drop=True)
        # update the track id of all tracks
        self.track['TRACK_ID'] = self.track.iloc[-1]['TRACK_ID']

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    """
    Customized implementation of BYTE Tracker for Tropical Cyclones Detection and Tracking.

    """
    def __init__(self, 
                 track_thresh, 
                 track_buffer, 
                 match_thresh, 
                 lats, 
                 lons, 
                 mot20 = False, 
                 frame_rate=30, # time-steps per second. Must always be 1 for TC Detection
                 ratio=30,      # Ratio for max_time_lost. Must always be 1 for TC Detection
                 ):
        self.lats = lats
        self.lons = lons
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.mot20 = mot20

        self.frame_id = 0
        self.det_thresh = track_thresh + 0.1
        self.buffer_size = int(frame_rate / ratio * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size, date = None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.lats, self.lons) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track: STrack = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, date)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, date=date)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.lats, self.lons) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, date)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, date=date)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, date)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id, date)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks
    
    def create_tracks_dataframe(self):
        if len(self.tracked_stracks) != 0:
            tracked_stracks_df = pd.concat([
                track.track for track in self.tracked_stracks
            ])
        else:
            tracked_stracks_df = pd.DataFrame(data={'ISO_TIME': [], 'TRACK_ID': [], 'LAT': [], 'LON': [], 'BBOX': []})
        tracked_stracks_df['TTYPE'] = 'tracked'
        
        if len(self.removed_stracks) != 0:
            removed_stracks_df = pd.concat([
                track.track for track in self.removed_stracks
            ])
        else:
            removed_stracks_df = pd.DataFrame(data={'ISO_TIME': [], 'TRACK_ID': [], 'LAT': [], 'LON': [], 'BBOX': []})
        removed_stracks_df['TTYPE'] = 'removed'
        
        if len(self.lost_stracks) != 0:
            lost_stracks_df = pd.concat([
                track.track for track in self.lost_stracks
            ])
        else:
            lost_stracks_df = pd.DataFrame(data={'ISO_TIME': [], 'TRACK_ID': [], 'LAT': [], 'LON': [], 'BBOX': []})
        lost_stracks_df['TTYPE'] = 'lost'
        
        stracks_df = pd.concat([tracked_stracks_df, removed_stracks_df, lost_stracks_df])
        # sort the dataframe by time
        stracks_df = stracks_df.sort_values(by='ISO_TIME')
        # drop the duplicates to avoid multiple instances of the same TC
        # stracks_df = stracks_df.drop_duplicates()
        # reset the index
        stracks_df = stracks_df.reset_index(drop=True)
        return stracks_df



def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
