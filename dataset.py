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
        Cached data @ https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/2.
        """
        self.transforms = [
            Transform().slowfast_transform,
            Transform().resnet_transform,
        ]
        self.root_dir = root_dir
        pkl_files = sorted(glob.glob(
            os.path.join(self.root_dir, 'Dataset_Information/Pkl_Files/*.pkl')
        ))
        self.annotations = load_annotations(pkl_files)
        self.normalization_parameters = qos_norm_params(self.annotations), labels_norm_params(self.annotations)
        self.max_duration = math.ceil(max([l['video_duration_sec'] for l in self.annotations]))
        cfg.T = 10

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        print("Loading video:", idx)
        video_path = os.path.join(self.root_dir, 'assets_mp4_individual', self.annotations[idx]['distorted_mp4_video'])
        video = EncodedVideo.from_path(video_path)

        duration = self.annotations[idx]['video_duration_sec']
        start_sec = 0
        end_sec = min(cfg.T, duration)

        video_clips = []

        for i in range(0, math.ceil(duration), cfg.T):
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

        labels_norm_params = self.normalization_parameters[1]

        overall_qoe = torch.tensor((self.annotations[idx]['retrospective_zscored_mos'] - labels_norm_params[1]) / (labels_norm_params[0] - labels_norm_params[1]))
        continuous_qoe = (
            (torch.tensor(self.annotations[idx]['continuous_zscored_mos']) - labels_norm_params[3]) / (labels_norm_params[2] - labels_norm_params[3])
        )[::math.ceil(self.annotations[idx]['frame_rate'])]

        if continuous_qoe.shape[0] != video_clips[1].shape[1]:
            print("\nWARNING: Continuous QoE label shape does not match video clips shape. Continuous QoE shape:", continuous_qoe.shape, "Video clips shape:", video_clips[1].shape, "\n")
            last = (
                (torch.tensor(self.annotations[idx]['continuous_zscored_mos']) - labels_norm_params[3]) / (labels_norm_params[2] - labels_norm_params[3])
            )[-1]
            continuous_qoe = torch.cat((continuous_qoe, last.unsqueeze(0)), dim=0)

        x = {
            'video_content': video_clips,
            'qos': qos_features,
            'overall_QoE': overall_qoe,
            'continuous_QoE': continuous_qoe,
        }

        return x


class PickleDataset(VideoDataset):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.directory = os.path.join(root_dir, 'assets_cached')
        self.dataset = VideoDataset(root_dir)
        self.cache_dataset()

    def cache_dataset(self):
        print("Caching dataset...")
        cached = []
        for i in range(self.__len__()):
            filename = os.path.join(self.directory, f"{i}.pt")
            if os.path.exists(filename):
                cached.append(i)

        for i in range(self.__len__()):
            if i not in cached:
                filename = os.path.join(self.directory,  f"{i}.pt")
                data = self.dataset[i]
                torch.save(data, filename)
                print("Video", i, "cached to", filename)

            else:
                print("Video", i, "already cached")

    def __len__(self):
        return len(glob.glob(self.directory + '/*.pt'))

    def __getitem__(self, idx):
        filename = os.path.join(self.directory, f"{idx}.pt")
        if os.path.exists(filename):
            return torch.load(filename)
        else:
            raise FileNotFoundError(f"Video {filename} not found. Please cache the dataset first.")