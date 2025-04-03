import torch.nn as nn   


class CrossFeatureAttention(nn.Module):
    """
    Implements the cross feature attention module for the dual-stage attention model.
    """
    def __init__(self):
        super(CrossFeatureAttention, self).__init__()
        self.d =  45  # n.features / n.groups = 180 / 4
        self.MSA = nn.MultiheadAttention(
            embed_dim=self.d,
            num_heads=3,
            batch_first=False  # Input shape: (seq_len, batch, embed_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (T, n_features)
        Returns:
            Output tensor of shape (T, n_features)
        """
        T = x.shape[0]
        
        x_reshaped = x.view(T, 4, self.d).permute(1, 0, 2)

        attn, _ = self.MSA(x_reshaped, x_reshaped, x_reshaped)
        
        output = attn.permute(1, 0, 2).contiguous().view(T, -1)
        
        return output