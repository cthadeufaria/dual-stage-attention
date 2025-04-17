import torch.nn as nn
from time import time

from backbone import Backbone
from feature_fusion import FeatureFusion
from fully_connected_networks import FC1
from short_time_regression import Simple1DCNN, Group1DCNN
from long_time_regression import LongTimeRegression
from cross_feature_attention import CrossFeatureAttention


class DualAttention(nn.Module):
    def __init__(self, device):
        super(DualAttention, self).__init__()
        self.device = device

        # Instantiate sub-networks.
        self.modules_dict = nn.ModuleDict({
            'backbone': Backbone(),
            'fc1': FC1(),
            'str_A': Simple1DCNN(),
            'str_B': Group1DCNN(),
            'cfa': CrossFeatureAttention(),
            'ltr_A': LongTimeRegression(1),
            'ltr_B': LongTimeRegression(2),
            'ff': FeatureFusion(),
        })

        self.to(device)

        self.modules_dict['backbone'] = self.modules_dict['backbone'].to('cpu')

    def forward(self, x):
        video_content_inputs, qos_features = x

        # Video content sub-network forward pass.
        print("Processing features with input shape:", (video_content_inputs[0][0].shape, video_content_inputs[0][1].shape), video_content_inputs[1].shape)

        video_content_features = self.modules_dict['backbone'](video_content_inputs)
        video_content_features = video_content_features.to(self.device)
        
        print("Video content features shape:", video_content_features.shape)

        downsampled_features = self.modules_dict['fc1'](video_content_features)
        temporal_reasoning_features = self.modules_dict['str_A'](downsampled_features)
        video_contents_attention_map = self.modules_dict['ltr_A'](temporal_reasoning_features)

        # QoS sub-network forward pass.
        qos_temporal_reasoning = self.modules_dict['str_B'](qos_features.squeeze(0))
        group_relations = self.modules_dict['cfa'](qos_temporal_reasoning)
        qos_attention_map = self.modules_dict['ltr_B'](group_relations)

        # Fuse video content and QoS sub-networks.
        fused_features = self.modules_dict['ff']((video_contents_attention_map, qos_attention_map))
        
        return fused_features