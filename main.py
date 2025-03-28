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
    Unsupportted GPU Github issue @ https://github.com/ROCm/rocBLAS/issues/1352.
    GPU support and compatibility matrices @
    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#linux-supported-gpus
    https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/native_linux/native_linux_compatibility.html
    https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html#architecture-support-compatibility-matrix
    """ 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # TODO: try to install rocm 5.3 to support compatibility with gfx1012 and AMD ATI Radeon RX 5500/5500M / Pro 5500M.
    device = torch.device('cpu')
    print(f"Device: {device}")
    
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
    dual_attention['ltr_A'] = LongTimeRegression(6).to(device)

    for net in dual_attention.values():
        net.eval()

    # Forward pass.
    video_content_features = dual_attention['backbone'](inputs)
    downsampled_features = dual_attention['fc1'](video_content_features)
    temporal_reasoning_features = dual_attention['str_A'](downsampled_features[None, :])
    attention_map = dual_attention['ltr_A'](temporal_reasoning_features)

    print(video_content_features.shape)
    print(downsampled_features.shape)
    print(temporal_reasoning_features.shape)
    print(attention_map.shape)
    

if __name__ == "__main__":
    main()