import os, glob, math, torch
from torch.utils.data import Dataset
from transforms import Transform
from pytorchvideo.data.encoded_video import EncodedVideo
from config import Config as cfg
from utils import load_annotations, batch_tensor, get_qos_features, qos_norm_params, labels_norm_params


class VideoDataset(Dataset):
    def __init__(self, root_dir):
        """
        Implementation of Dataset class for LIVE NETFLIX - II dataset.
        More info found @ http://live.ece.utexas.edu/research/LIVE_NFLX_II/live_nflx_plus.html
        and @ http://live.ece.utexas.edu/research/LIVE_NFLXStudy/nflx_index.html.
        """
        self.transforms = [
            Transform().slowfast_transform,
            Transform().resnet_transform,
        ]
        self.root_dir = root_dir
        pkl_files = glob.glob(
            os.path.join(self.root_dir, 'Dataset_Information/Pkl_Files/*.pkl')
        )
        self.annotations = load_annotations(pkl_files)
        self.normalization_parameters = qos_norm_params(self.annotations), labels_norm_params(self.annotations)
        # TODO: select max tensor size of video clips here.
        cfg.T = 10

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, 'assets_mp4_individual', self.annotations[idx]['distorted_mp4_video'])
        video = EncodedVideo.from_path(video_path)

        duration = self.annotations[idx]['video_duration_sec']
        start_sec = 0
        end_sec = min(cfg.T, duration)

        video_clips = []

        print("Parsing", round(duration, 2), "seconds of video", idx, 'from dataset')

        for i in range(0, math.ceil(duration), cfg.T):
            print("Processing chunk", int(i/cfg.T) + 1, 'of', math.ceil(math.ceil(duration)/cfg.T))

            slowfast_transform = self.transforms[0]((end_sec - start_sec) * cfg.slowfast_sample_size, cfg.downsample_size, cfg.mean, cfg.std)
            resnet_transform = self.transforms[1](cfg.mean, cfg.std, (end_sec - start_sec) * cfg.resnet_sample_size)

            slowfast_clip = slowfast_transform(video.get_clip(start_sec=start_sec, end_sec=end_sec))['video']
            resnet_clip = resnet_transform(video.get_clip(start_sec=start_sec, end_sec=end_sec))['video']
            
            video_clips.append((slowfast_clip, resnet_clip))

            start_sec += cfg.T
            end_sec += cfg.T
            end_sec = min(math.ceil(duration), end_sec)

        qos_features = get_qos_features(self.annotations[idx])

        max_values = self.normalization_parameters[0][0]
        min_values = self.normalization_parameters[0][1]

        qos_features[:, 0] = (qos_features[:, 0] - min_values['playback_indicator']) / (max_values['playback_indicator'] - min_values['playback_indicator'])
        qos_features[:, 1] = (qos_features[:, 1] - min_values['temporal_recency_feature']) / (max_values['temporal_recency_feature'] - min_values['temporal_recency_feature'])
        qos_features[:, 2] = (qos_features[:, 2] - min_values['representation_quality']) / (max_values['representation_quality'] - min_values['representation_quality'])
        qos_features[:, 3] = (qos_features[:, 3] - min_values['bitrate_switch']) / (max_values['bitrate_switch'] - min_values['bitrate_switch'])

        video_clips = batch_tensor(video_clips)
        # TODO: pad video clips to max size of video clips.
        
        labels_norm_params = self.normalization_parameters[1]

        overall_qoe = torch.tensor((self.annotations[idx]['retrospective_zscored_mos'] - labels_norm_params[1]) / (labels_norm_params[0] - labels_norm_params[1]))
        continuous_qoe = (
            (torch.tensor(self.annotations[idx]['continuous_zscored_mos']) - labels_norm_params[3]) / (labels_norm_params[2] - labels_norm_params[3])
        )[::math.ceil(self.annotations[idx]['frame_rate'])]

        print("Video", idx, "continuous QoE label shape:", continuous_qoe.shape)

        if continuous_qoe.shape[0] != video_clips[1].shape[1]:
            print("\nWARNING: Continuous QoE label shape does not match video clips shape. Continuous QoE shape:", continuous_qoe.shape, "Video clips shape:", video_clips[1].shape)

        return {
        'video_content': video_clips,  # (slowfast, resnet)
        'qos': qos_features,  # (num_timesteps, qos_features)
        'overall_QoE': overall_qoe,
        'continuous_QoE': continuous_qoe,
        }