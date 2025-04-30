import torch
import torch.nn as nn
from torch import randn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class LongTimeRegression(nn.Module):
    """
    Implements the long-time regression network for the dual-stage attention model.
    Multi-Head Masked Attention implementation described @ 
    https://www.geeksforgeeks.org/how-to-use-pytorchs-nnmultiheadattention/#handling-masks-with-nnmultiheadattention.
    
    src_key_padding_mask implementation described @ https://iori-yamahata.net/2024/02/28/programming-1-eng/
    and @ https://sanjayasubedi.com.np/deeplearning/masking-in-attention/.
    """
    def __init__(self, padding, num_layers, model_dim=180, n_head=4):
        super(LongTimeRegression, self).__init__()

        self.position_encoder = nn.Parameter(randn(padding, model_dim))

        encoder_layer = TransformerEncoderLayer(
            model_dim,
            n_head,
            dim_feedforward=4 * model_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.FC = nn.Linear(180, 180)
        self.activation = nn.SELU()

    def forward(self, x):
        """
        Input shape: (seq_len, d_model)
        Output shape: (seq_len, d_model)
        """
        seq_len = x.shape[0]

        F = x + self.position_encoder[:seq_len, :]

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(F.device)

        C = self.transformer_encoder(
            F,
            mask=causal_mask,
        )

        O = self.activation(self.FC(C)) + C

        return O