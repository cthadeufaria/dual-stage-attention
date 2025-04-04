import torch.nn as nn

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
        
        # Instantiate sub-networks. # TODO: implement a class to encapsulate the dual attention model and instantiate in main.
        self.modules = {}
        self.modules['backbone'] = Backbone().to(device)
        self.modules['fc1'] = FC1().to(device)
        self.modules['str_A'] = Simple1DCNN().to(device)
        self.modules['str_B'] = Group1DCNN().to(device)
        self.modules['cfa'] = CrossFeatureAttention().to(device)
        self.modules['ltr_A'] = LongTimeRegression(1).to(device)
        self.modules['ltr_B'] = LongTimeRegression(2).to(device)
        self.modules['ff'] = FeatureFusion().to(device)
    
    def forward(self, x):
        video_content_inputs, qos_features = x

        # Video content sub-network forward pass.
        video_content_features = self.modules['backbone'](video_content_inputs)
        downsampled_features = self.modules['fc1'](video_content_features)
        temporal_reasoning_features = self.modules['str_A'](downsampled_features[None, :])
        video_contents_attention_map = self.modules['ltr_A'](temporal_reasoning_features)

        # QoS sub-network forward pass.
        qos_temporal_reasoning = self.modules['str_B'](qos_features)
        group_relations = self.modules['cfa'](qos_temporal_reasoning)
        qos_attention_map = self.modules['ltr_B'](group_relations)

        # Fuse video content and QoS sub-networks.
        fused_features = self.modules['ff']((video_contents_attention_map, qos_attention_map))

        # Debug print
        print('\nSub-Network A\n')
        print(video_content_features.shape)
        print(downsampled_features.shape)
        print(temporal_reasoning_features.shape)
        print(video_contents_attention_map.shape)
        print('\nSub-Network B\n')
        print(qos_features.shape)
        print(qos_temporal_reasoning.shape)
        print(group_relations.shape)
        print(qos_attention_map.shape)
        print('\nOverall and Continuous QoE prediction:', fused_features[0].shape, fused_features[1].shape)
        
        return fused_features