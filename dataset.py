import os, pickle, torch, sys, types, glob, math
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
        self.annotations = self.load_annotations(pkl_files)
        self.T = 10

    def load_annotations(self, pkl_files):
        annotations = []
        for file in pkl_files:
            with open(file, 'rb') as f:
                annotations.append(pickle.load(f, encoding='latin1'))
                
        return annotations

    def preprocess(self, nested_list):
        merged_tensor1 = torch.cat([sublist[0][0] for sublist in nested_list], dim=1)
        merged_tensor2 = torch.cat([sublist[0][1] for sublist in nested_list], dim=1)        
        merged_tensor3 = torch.cat([sublist[1] for sublist in nested_list], dim=1)
        
        return [[merged_tensor1, merged_tensor2], merged_tensor3]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, 'assets_mp4_individual', self.annotations[idx]['distorted_mp4_video'])
        video = EncodedVideo.from_path(video_path)

        frame_rate = self.annotations[idx]['frame_rate']
        duration = self.annotations[idx]['video_duration_sec']
        start_sec = 0
        end_sec = min(self.T, duration)

        video_clips = []
        qos_features = []

        downsample_size = (224, 224)
        mean = [0.45, 0.45, 0.45] # TODO: check if normalization parameters are correct.
        std = [0.225, 0.225, 0.225]
        slowfast_sample_size = 32
        resnet_sample_size = 1

        print("Retrieving video", idx, 'from dataset...')

        for i in range(0, math.ceil(duration), self.T):
            print('Transforming video clip', idx, '.', i, '...')

            slowfast_transform = self.transforms[0]((end_sec - start_sec) * slowfast_sample_size, downsample_size, mean, std)
            resnet_transform = self.transforms[1](mean, std, (end_sec - start_sec) * resnet_sample_size)

            slowfast_clip = slowfast_transform(video.get_clip(start_sec=start_sec, end_sec=end_sec))['video']
            resnet_clip = resnet_transform(video.get_clip(start_sec=start_sec, end_sec=end_sec))['video']
            
            video_clips.append((slowfast_clip, resnet_clip))

            for i in range(start_sec, end_sec):
                start_frame = int(start_sec * frame_rate)
                end_frame = int(end_sec * frame_rate)

                last_start_frame = int(max(0, (start_sec - self.T)) * frame_rate)

                playback_indicator = torch.tensor(sum(self.annotations[idx]['is_rebuffered_bool'][start_frame : end_frame]) / frame_rate)

                rebuffered = self.annotations[idx]['is_rebuffered_bool'][:end_frame]

                ones_indices = [i for i, x in enumerate(rebuffered) if x == 1]

                if not ones_indices:
                    temporal_recency_feature = len(rebuffered) / (frame_rate * duration)
                else:
                    last_one_idx = ones_indices[-1]
                    temporal_recency_feature = len(rebuffered) - last_one_idx - 1 / (frame_rate * duration)
                
                temporal_recency_feature = torch.tensor(temporal_recency_feature)

                avg_bitrate = avg(self.annotations[idx]['playout_bitrate'][start_frame : end_frame])
                epsilon = 1e-6  # Prevents log(0)
                representation_quality = torch.tensor(np.float32(np.log10(avg_bitrate + epsilon)))
        
                if start_sec == 0:
                    bitrate_switch = torch.tensor(0)

                else:
                    bitrate_switch = torch.tensor(np.float32(max(0, 
                        avg(self.annotations[idx]['playout_bitrate'][start_frame : end_frame]) -
                        avg(self.annotations[idx]['playout_bitrate'][last_start_frame : start_frame])
                    )))

                qos_features.append(torch.stack(
                    [playback_indicator, temporal_recency_feature, representation_quality, bitrate_switch]
                ))

            start_sec += self.T
            end_sec += self.T
            end_sec = min(math.ceil(duration), end_sec)

        qos_features = torch.stack(qos_features)
        video_clips = self.preprocess(video_clips)

        print('QoS features shape:', qos_features.shape)
        print('Video clips shape:', len(video_clips), video_clips[0][0].shape, video_clips[0][1].shape, video_clips[1].shape)

        return {
        'video_content': video_clips,  # List of (slowfast, resnet) transformed clips
        'qos': qos_features  # Tensor of shape (num_timesteps, num_features)
        }