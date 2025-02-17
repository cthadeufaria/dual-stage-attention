import torch
from torch.utils.data import DataLoader
from dataset import VideoDataset
from backbone import Backbone
from fully_connected_networks import FC1, FC2, FC3, FC4, FC5
from short_time_regression import Simple1DCNN, Group1DCNN


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # One video chunk of 1s.
    video_chunk = 1

    # Load the dataset.
    inputs = [
        [a[0].to(device), a[1].to(device)] if type(a) == list else a.to(device)
        for a in next(iter(DataLoader(VideoDataset('./datasets/LIVE_NFLX_Plus', video_chunk))))
    ]

    # Instantiate sub-networks.
    dual_attention = {}
    dual_attention['backbone'] = Backbone().to(device)
    dual_attention['fc1'] = FC1().to(device)
    dual_attention['str_B'] = Group1DCNN().to(device)
    dual_attention['str_A'] = Simple1DCNN().to(device)

    for net in dual_attention.values():
        net.eval()

    # Forward pass.
    video_content_features = dual_attention['backbone'](inputs)
    downsampled_features = dual_attention['fc1'](video_content_features)
    temporal_reasoning_features = dual_attention['str_A'](downsampled_features[None, :])

    print(temporal_reasoning_features.shape)
    

if __name__ == "__main__":
    main()