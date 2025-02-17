import os, pickle, torch, sys, types, glob
from torch.utils.data import Dataset
import torch.nn as nn
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

    def __getitem__(self, idx): # TODO: get QoS features from the dataset.
        downsample_size = (224, 224)
        mean = [0.45, 0.45, 0.45] # TODO: check if normalization parameters are correct.
        std = [0.225, 0.225, 0.225]
        slowfast_sample_size = 32
        resnet_sample_size = 1

        video_path = os.path.join(self.root_dir, 'assets_mp4_individual', self.annotations[idx]['distorted_mp4_video'])
        video = EncodedVideo.from_path(video_path)
        video_data = [
            video.get_clip(start_sec=0, end_sec=self.T),
            video.get_clip(start_sec=0, end_sec=self.T)
        ]

        slowfast_transform = self.transforms[0](self.T * slowfast_sample_size, downsample_size, mean, std)
        resnet_transform = self.transforms[1](mean, std, self.T * resnet_sample_size)

        slowfast_transform(video_data[0])
        resnet_transform(video_data[1])

        return [v['video'] for v in video_data]