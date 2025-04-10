import pickle
import torch
import math
import numpy as np


def load_annotations(pkl_files):
    annotations = []
    for file in pkl_files:
        with open(file, 'rb') as f:
            annotations.append(pickle.load(f, encoding='latin1'))

    return annotations

def batch_tensor(nested_list):
    """
    Concatenates sublists to form main tensor.
    """
    tensor1 = torch.cat([sublist[0][0] for sublist in nested_list], dim=1)
    tensor2 = torch.cat([sublist[0][1] for sublist in nested_list], dim=1)        
    tensor3 = torch.cat([sublist[1] for sublist in nested_list], dim=1)
    
    return [[tensor1, tensor2], tensor3]

def labels_norm_params(annotations):
    """
    Returns the maximum and minimum values of the retrospective 
    and continuous MOS over the whole dataset.
    """
    r_max_value = -1000
    r_min_value = 1000
    c_max_value = -1000
    c_min_value = 1000

    for annotation in annotations:
        if annotation['retrospective_zscored_mos'] > r_max_value:
            r_max_value = annotation['retrospective_zscored_mos']
        if annotation['retrospective_zscored_mos'] < r_min_value:
            r_min_value = annotation['retrospective_zscored_mos']

        if max(annotation['continuous_zscored_mos']) > c_max_value:
            c_max_value = max(annotation['continuous_zscored_mos'])
        if min(annotation['continuous_zscored_mos']) < c_min_value:
            c_min_value = min(annotation['continuous_zscored_mos'])

    return r_max_value, r_min_value, c_max_value, c_min_value

def qos_norm_params(annotations):
    """
    Returns the maximum and minimum values of the QoS features.
    """
    max_qos = {
        'playback_indicator': -1000,
        'temporal_recency_feature': -1000,
        'representation_quality': -1000,
        'bitrate_switch': -1000
    }
    min_qos = {
        'playback_indicator': 1000,
        'temporal_recency_feature': 1000,
        'representation_quality': 1000,
        'bitrate_switch': 1000
    }

    for annotation in annotations:
        qos_features = get_qos_features(annotation)

        if max_qos['playback_indicator'] < max(qos_features[:, 0]):
            max_qos['playback_indicator'] = max(qos_features[:, 0])
        if min_qos['playback_indicator'] > min(qos_features[:, 0]):
            min_qos['playback_indicator'] = min(qos_features[:, 0])
        if max_qos['temporal_recency_feature'] < max(qos_features[:, 1]):
            max_qos['temporal_recency_feature'] = max(qos_features[:, 1])
        if min_qos['temporal_recency_feature'] > min(qos_features[:, 1]):
            min_qos['temporal_recency_feature'] = min(qos_features[:, 1])
        if max_qos['representation_quality'] < max(qos_features[:, 2]):
            max_qos['representation_quality'] = max(qos_features[:, 2])
        if min_qos['representation_quality'] > min(qos_features[:, 2]):
            min_qos['representation_quality'] = min(qos_features[:, 2])
        if max_qos['bitrate_switch'] < max(qos_features[:, 3]):
            max_qos['bitrate_switch'] = max(qos_features[:, 3])
        if min_qos['bitrate_switch'] > min(qos_features[:, 3]):
            min_qos['bitrate_switch'] = min(qos_features[:, 3])
    
    return max_qos, min_qos

def get_qos_features(annotation):
    """
    Returns the QoS features for each video.
    """
    qos_features = []

    frame_rate = annotation['frame_rate']
    duration = annotation['video_duration_sec']
    start_sec = 0
    end_sec = 1

    for i in range(0, math.ceil(duration)):
        start_frame = int(start_sec * frame_rate)
        end_frame = int(end_sec * frame_rate)

        last_start_frame = int(max(0, (start_sec - 1)) * frame_rate)

        playback_indicator = torch.tensor(sum(annotation['is_rebuffered_bool'][start_frame : end_frame]) / frame_rate)

        rebuffered = annotation['is_rebuffered_bool'][:end_frame]

        ones_indices = [i for i, x in enumerate(rebuffered) if x == 1]

        if not ones_indices:
            temporal_recency_feature = len(rebuffered) / (frame_rate * duration)
        else:
            last_one_idx = ones_indices[-1]
            temporal_recency_feature = len(rebuffered) - last_one_idx - 1 / (frame_rate * duration)

        temporal_recency_feature = torch.tensor(temporal_recency_feature)

        avg_bitrate = np.average(annotation['playout_bitrate'][start_frame : end_frame])
        epsilon = 1e-6  # Prevents log(0)
        representation_quality = torch.tensor(np.float32(np.log10(avg_bitrate + epsilon)))

        if start_sec == 0:
            bitrate_switch = torch.tensor(0)
        else:
            bitrate_switch = torch.tensor(np.float32(max(0, 
                np.average(annotation['playout_bitrate'][start_frame : end_frame]) -
                np.average(annotation['playout_bitrate'][last_start_frame : start_frame])
            )))

        qos_features.append(torch.stack(
            [playback_indicator, temporal_recency_feature, representation_quality, bitrate_switch]
        ))

        start_sec += 1
        end_sec += 1

    return torch.stack(qos_features)