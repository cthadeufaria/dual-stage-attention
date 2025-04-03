import torch
from torch.utils.data import DataLoader
from dataset import VideoDataset
from backbone import Backbone
from feature_fusion import FeatureFusion
from fully_connected_networks import FC1
from short_time_regression import Simple1DCNN, Group1DCNN
from long_time_regression import LongTimeRegression
from cross_feature_attention import CrossFeatureAttention


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
    
    # Define video chunk size.
    video_chunk = 1 # seconds

    # Load the dataset.
    inputs = [
        [a[0].to(device), a[1].to(device)] if type(a) == list else a.to(device) 
        for a in next(iter(DataLoader(VideoDataset('./datasets/LIVE_NFLX_Plus', video_chunk))))
    ]

    video_content_inputs = inputs[:2]
    qos_features = torch.tensor(inputs[2:]).to(device)

    # Instantiate sub-networks.
    dual_attention = {}
    dual_attention['backbone'] = Backbone().to(device)
    dual_attention['fc1'] = FC1().to(device)
    dual_attention['str_A'] = Simple1DCNN().to(device)
    dual_attention['str_B'] = Group1DCNN().to(device)
    # dual_attention['cfa'] = CrossFeatureAttention().to(device)
    dual_attention['ltr_A'] = LongTimeRegression(1).to(device)
    dual_attention['ltr_B'] = LongTimeRegression(2).to(device)
    dual_attention['ff'] = FeatureFusion().to(device)

    for net in dual_attention.values():
        net.eval()

    # Video content sub-network forward pass.
    video_content_features = dual_attention['backbone'](video_content_inputs)
    downsampled_features = dual_attention['fc1'](video_content_features)
    temporal_reasoning_features = dual_attention['str_A'](downsampled_features[None, :])
    video_contento_attention_map = dual_attention['ltr_A'](temporal_reasoning_features)

    # QoS sub-network forward pass. # TODO: implement a class to encapsulate the dual attention model and instantiate in main.
    qos_temporal_reasoning = dual_attention['str_B'](qos_features)
    group_relations = dual_attention['cfa'](qos_temporal_reasoning)
    qos_attention_map = dual_attention['ltr_B'](group_relations)

    # Fuse video content and QoS sub-networks.
    fused_features = dual_attention['ff']((video_contento_attention_map, qos_attention_map))

    print(video_content_features.shape)
    print(downsampled_features.shape)
    print(temporal_reasoning_features.shape)
    print(video_contento_attention_map.shape)
    print(qos_temporal_reasoning.shape)
    # print(group_relations.shape)
    print(qos_attention_map.shape)
    print(fused_features.shape)
    

if __name__ == "__main__":
    main()