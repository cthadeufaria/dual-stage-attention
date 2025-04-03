import torch.nn as nn


class Simple1DCNN(nn.Module):
    """
    Implements a simple 1D-CNN model for the Short-Time Temporal Regression Module
    in the Video Content Feature Processing Sub-Network.
    No need for max pooling here as the output must have the same length as the input.
    Architecture follows proposal used in https://doi.org/10.3390/app14188500.
    """
    def __init__(self):
        super(Simple1DCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ZeroPad1d((4, 0)),
            nn.Conv1d(
                in_channels=1, 
                out_channels=1, 
                kernel_size=5
            ),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.ZeroPad1d((4, 0)),
            nn.Conv1d(
                in_channels=1, 
                out_channels=1, 
                kernel_size=5
            ),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Flatten(0),
            nn.Linear(
                in_features=180, 
                out_features=180
            ),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class Group1DCNN(nn.Module):
    """ 
    Implements a group 1D-CNN model for the Short-Time Temporal Regression Module
    in the QoS Feature Processing Sub-Network.
    No need for max pooling here as the output must have the same length as the input.
    Architecture follows proposal used in https://doi.org/10.3390/app14188500.
    """
    def __init__(self):
        super(Group1DCNN, self).__init__()
        
        # Layer 1: (1,4,T) → (1,180,T)
        self.layer1 = nn.Sequential(
            nn.ZeroPad1d((4, 0)),
            nn.Conv1d(
                in_channels=4, 
                out_channels=180, 
                kernel_size=5, 
                groups=4, 
            ),
            nn.ReLU()
        )
        
        # Layer 2: (1,180,T) → (1,180,T)
        self.layer2 = nn.Sequential(
            nn.ZeroPad1d((4, 0)),
            nn.Conv1d(
                in_channels=180, 
                out_channels=180, 
                kernel_size=5, 
                groups=4, 
            ),
            nn.ReLU()
        )

    def forward(self, x):
        # Input shape: (T, 4)
        x = x[None, :].permute(1, 0).unsqueeze(0)  # → (1, 4, T)
        x = self.layer1(x)                 # → (1, 180, T)
        x = self.layer2(x)                 # → (1, 180, T)
        x = x.permute(0, 2, 1).squeeze(0) # → (T, 180)

        return x