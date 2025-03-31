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
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.ZeroPad1d((4, 0)),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Flatten(0),
            nn.Linear(in_features=180, out_features=180),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class Group1DCNN(nn.Module): # TODO: Validate architecture. How input.shape = (T, 4) and output.shape = (T, 180)? Define QoS features shape.
    """ 
    Implements a group 1D-CNN model for the Short-Time Temporal Regression Module
    in the QoS Feature Processing Sub-Network.
    No need for max pooling here as the output must have the same length as the input.
    Architecture follows proposal used in https://doi.org/10.3390/app14188500.
    """
    def __init__(self):
        super(Group1DCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ZeroPad1d((4, 0)),
            nn.Conv1d(in_channels=1, out_channels=1, groups=4, kernel_size=5),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.ZeroPad1d((4, 0)),
            nn.Conv1d(in_channels=1, out_channels=1, groups=4, kernel_size=5),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Flatten(0),
            nn.Linear(in_features=180, out_features=180),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x