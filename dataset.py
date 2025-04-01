import os, pickle, torch, sys, types, glob
from torch.utils.data import Dataset
from numpy import average as avg
import torch.nn as nn
import numpy as np
from torchvision.transforms import (
    Compose, 
    Lambda,
    Resize,
)
from torchvision.transforms._transforms_video import NormalizeVideo
# Fixed module import error using the impl. @ https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985#issuecomment-2439896362
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms.functional import rgb_to_grayscale
functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
)


class Transform:
    """
    Class to define the transformation of the video input for the model and make it reusable.
    """
    def __init__(self):
        pass

    def slowfast_transform(self, T, downsample_size, mean, std):
        return  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    # TODO: Must be first 24 frames in each second? Do not work with uniformly separated samples?
                    UniformTemporalSubsample(T),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    Resize(downsample_size),
                    PackPathway()
                ]
            ),
        )

    def resnet_transform(self, mean, std, T):
        return  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(T),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                ]
            ),
        )


class PackPathway(nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor, slowfast_alpha: int = 4):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        
        return frame_list


class VideoDataset(Dataset):
    def __init__(self, root_dir, timestep):
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
        self.annotations = self.load_annotations(pkl_files)
        self.T = timestep

    def load_annotations(self, pkl_files):
        annotations = []
        for file in pkl_files:
            with open(file, 'rb') as f:
                annotations.append(pickle.load(f, encoding='latin1'))
                
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        downsample_size = (224, 224)
        mean = [0.45, 0.45, 0.45] # TODO: check if normalization parameters are correct.
        std = [0.225, 0.225, 0.225]
        slowfast_sample_size = 32
        resnet_sample_size = 1

        start_sec = 1
        end_sec = self.T + 1

        frame_rate = self.annotations[idx]['frame_rate']
        start_frame = int(start_sec * frame_rate)
        end_frame = int(end_sec * frame_rate)
        last_start_frame = int(max(0, (self.T - 1)) * frame_rate)

        video_path = os.path.join(self.root_dir, 'assets_mp4_individual', self.annotations[idx]['distorted_mp4_video'])
        video = EncodedVideo.from_path(video_path)

        playback_indicator = torch.tensor(sum(self.annotations[idx]['is_rebuffered_bool'][start_frame : end_frame]) / frame_rate)

        rebuffered = self.annotations[idx]['is_rebuffered_bool'][:end_frame]

        ones_indices = [i for i, x in enumerate(rebuffered) if x == 1]

        duration = self.annotations[idx]['video_duration_sec']

        if not ones_indices:
            temporal_recency_feature = len(rebuffered) / (frame_rate * duration)
        else:
            last_one_idx = ones_indices[-1]
            temporal_recency_feature = len(rebuffered) - last_one_idx - 1 / (frame_rate * duration)
        
        temporal_recency_feature = torch.tensor(temporal_recency_feature)

        avg_bitrate = torch.tensor(avg(self.annotations[idx]['playout_bitrate'][start_frame : end_frame]))
        epsilon = 1e-6  # Prevents log(0)
        representation_quality = torch.tensor(np.log10(avg_bitrate + epsilon))

        if start_sec == 0:
            bitrate_switch = torch.tensor(0)

        else:
            bitrate_switch = torch.tensor(max(0, 
                avg(self.annotations[idx]['playout_bitrate'][start_frame : end_frame]) -
                avg(self.annotations[idx]['playout_bitrate'][last_start_frame : start_frame])
            ))

        video_data = [
            video.get_clip(start_sec=start_sec, end_sec=end_sec),
            video.get_clip(start_sec=start_sec, end_sec=end_sec),
            playback_indicator,
            temporal_recency_feature,
            representation_quality,
            bitrate_switch,
        ]

        slowfast_transform = self.transforms[0](self.T * slowfast_sample_size, downsample_size, mean, std)
        resnet_transform = self.transforms[1](mean, std, self.T * resnet_sample_size)

        slowfast_transform(video_data[0])
        resnet_transform(video_data[1])

        return [v['video'] for v in video_data]