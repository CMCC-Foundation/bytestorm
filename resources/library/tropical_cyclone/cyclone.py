# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The GPLv3 License [see LICENSE for details]#
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from tropical_cyclone._cyclone.tracking import init_track_dataframe, tracking_algorithm
from tropical_cyclone._cyclone.patch import (
    get_all_adjacent_patches,
    get_nearest_adjacent_patches,
    get_nocyclones_patches,
    get_random_patches,
)
from tropical_cyclone._cyclone.macros import DENSITY_MAP_TC, LABEL_MAP_TC, SQUARE_MAP_TC, CMCC_CM3_VARS, ERA5_CMIP6_DRIVERS_MAPPING
from tropical_cyclone._cyclone.augmentation import coo_left_right, coo_rot180, coo_up_down, rot180, left_right, up_down
from tropical_cyclone._cyclone.georef import from_local_to_global, get_global_rowcol_coords
from tropical_cyclone._cyclone.utils import (
    compute_euclidean_distance,
    get_ibtracs_by_basins,
    classification_score,
    get_tc_map_labels,
    get_tropical_cyclone_positions,
    build_tc_filenames,
    retrieve_predicted_tc,
    filter_basin,
    filter_tracks,
    compute_pod_and_far,
    compute_storm_transits,
    signif,
    mean,
    std,
    generate_random_origin,
    generate_random_coordinate,
    parse_bboxes_from_str_to_array,
    remove_tcs_above_x_lat,
    remove_tcs_originated_on_land,
    remove_short_tracks,
    convert_coords_to_format,
    bearing,
    track_smoothness,
    clean_tracks,
    )
from tropical_cyclone._cyclone.inference import (
    Inference,
    SingleModelInference,
    LocClsModelInference,
    load_trained_model,
    prepare_dataloader,
    predict_with_models,
    get_detections,
    predict_with_models_cls,
    predict_with_models_clsloc,
    )
from tropical_cyclone._cyclone.visualize import (
    plot_detections,
    plot_tracks,
    plot_storm_track_density,
    plot_pod_and_far,
    plot_pod_and_far_multi_trackers,
    plot_trackers_biases_wrt_baseline,
    plot_pod_and_far_on_axis,
    )
from tropical_cyclone._cyclone.tracker.byte_tracker import BYTETracker