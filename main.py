import torch

from torch.utils.data import DataLoader
from dataset import VideoDataset
from dual_attention import DualAttention


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    dual_attention = DualAttention(device)
    
    video_chunk_A = 5  # seconds # TODO: use different values of T.
    video_chunk_B = 4  # seconds # TODO: test different for each sub-network.

    dataloader = DataLoader(VideoDataset('./datasets/LIVE_NFLX_Plus', video_chunk_A))

    inputs = next(iter(dataloader))
    
    video_content_inputs = []

    for chunk in inputs['video_content']:
        video_content_inputs.append([
            [b[0].to(device), b[1].to(device)] if type(b) == list else b.to(device) for b in chunk
        ])

        qos_features = torch.stack([a for a in inputs['qos']]).to(device)

    print(video_content_inputs[0][1].shape)
    print(video_content_inputs[0][0][0].shape)
    print(video_content_inputs[0][0][1].shape)

    for module in dual_attention.modules.values():
        module.eval()

    dual_attention((video_content_inputs, qos_features))


if __name__ == "__main__":
    main()