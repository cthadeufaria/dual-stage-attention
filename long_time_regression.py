import torch.nn as nn
from torch import randn, triu, ones
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class LongTimeRegression(nn.Module):
    """
    Implements the long-time regression network for the dual-stage attention model.
    Multi-Head Masked Attention implementation described @ 
    https://www.geeksforgeeks.org/how-to-use-pytorchs-nnmultiheadattention/#handling-masks-with-nnmultiheadattention.
    src_key_padding_mask implementation described @ https://iori-yamahata.net/2024/02/28/programming-1-eng/.
    """
    def __init__(self, num_layers=1, max_seq_len=1000, model_dim=180, n_head=4):
        super(LongTimeRegression, self).__init__()

        self.position_encoder = nn.Parameter(randn(max_seq_len, model_dim))

        encoder_layer = TransformerEncoderLayer(
            model_dim, 
            n_head, 
            dim_feedforward=4 * model_dim, 
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.FC = nn.Linear(180, 180)
        self.activation = nn.SELU()

    def forward(self, x):
        """
        Input shape: (batch_size, seq_len, d_model)
        Output shape: (batch_size, seq_len, d_model)
        """
        x = x[None, None, :] if len(x.shape) == 1 else x[None, :] if len(x.shape) == 2 else x
        seq_len = x.shape[1] # T video chunk
        # TODO: create src_key_padding_mask
        src_key_padding_mask = None

        F = x + self.position_encoder[None, :seq_len, :]  # (1, seq_len, d_model)
        F = F.permute(1, 0, 2) # (seq_len, 1, d_model)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(F.device) 

        C = self.transformer_encoder(F, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        # C = self.transformer_encoder(F, mask=causal_mask)
        C = C.permute(1, 0, 2) # Return to original shape (1, seq_len, d_model)

        O = self.activation(self.FC(C)) + C  # TODO: this should be the output of each encoder block. And there's many encoder blocks supposedly. Double check theory.

        return O.squeeze(0)