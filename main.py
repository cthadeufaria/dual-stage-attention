import torch
from torch.utils.data import DataLoader
from dataset import VideoDataset
from backbone import Backbone
from fully_connected_networks import FC1, FC2, FC3, FC4, FC5
from short_time_regression import Simple1DCNN, Group1DCNN
from long_time_regression import LongTimeRegression


def main():
    """
    Install PyTorch with ROCm Compute Platform using info @ https://pytorch.org/get-started/locally/.
    HIP and ROCm installation instructions for cuda impl. @ https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define video chunk size.
    video_chunk = 1 # seconds

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
    dual_attention['ltr_A'] = LongTimeRegression().to(device)

    for net in dual_attention.values():
        net.eval()

    # Forward pass.
    video_content_features = dual_attention['backbone'](inputs)
    downsampled_features = dual_attention['fc1'](video_content_features)
    temporal_reasoning_features = dual_attention['str_A'](downsampled_features[None, :])
    attention_map = dual_attention['ltr_A'](temporal_reasoning_features)

    print(attention_map.shape)
    

if __name__ == "__main__":
    main()