import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention


class LongTimeRegression(nn.Module):
    """
    Implements the long-time regression network for the dual-stage attention model.
    Detailed implementation of the Transformer Decoder for reference @ 
    https://medium.com/data-scientists-diary/implementation-of-transformer-encoder-in-pytorch-daeb33a93f9c.
    """
    def __init__(self):
        super(LongTimeRegression, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=180, nhead=6)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=6)

    def forward(self, x):
        x = self.transformer_encoder(x)

        return x