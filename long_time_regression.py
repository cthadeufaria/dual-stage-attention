import torch.nn as nn
from torch import randn, triu, ones
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class LongTimeRegression(nn.Module):
    """
    Implements the long-time regression network for the dual-stage attention model.
    Multi-Head Masked Attention implementation described @ 
    https://www.geeksforgeeks.org/how-to-use-pytorchs-nnmultiheadattention/#handling-masks-with-nnmultiheadattention.
    """
    def __init__(self, num_layers=1, max_seq_len=1000, model_dim=180, n_head=4):
        super(LongTimeRegression, self).__init__()

        self.position_encoder = nn.Parameter(randn(max_seq_len, model_dim))
        
        encoder_layer = TransformerEncoderLayer(
            model_dim, n_head, dim_feedforward=4 * model_dim, dropout=0.1, activation='relu'
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.FC = nn.Linear(180, 180)
        self.activation = nn.SELU()

    def generate_causal_mask(self, sz):
        """Generate causal mask to prevent attending to future positions"""
        return triu(ones(sz, sz) == 1).transpose(0, 1).float().masked_fill(
            triu(ones(sz, sz)) == 0, float('-inf'))
    
    def forward(self, x):
        """
        Input shape: (batch_size, seq_len, d_model)
        Output shape: (batch_size, seq_len, d_model)
        """
        x = x[None, None, :] if len(x.shape) == 1 else x[None, :] if len(x.shape) == 2 else x # TODO: implement correct input shape considering seq_len = T.
        seq_len = x.shape[1] # T / video chunk
        
        F = x + self.position_encoder[None, :seq_len, :]  # (1, seq_len, d_model)
        F = F.permute(1, 0, 2)
        
        causal_mask = self.generate_causal_mask(seq_len).to(F.device)

        C = self.transformer_encoder(F, mask=causal_mask)
        C = C.permute(1, 0, 2) # Return to original shape (batch_size, seq_len, d_model)

        O = self.activation(self.FC(C)) + C  # TODO: this should be the output of each encoder block. And there's many encoder blocks supposedly. Double check theory.

        return O.squeeze() # TODO: implement correct output shape.