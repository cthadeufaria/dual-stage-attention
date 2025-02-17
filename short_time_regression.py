import torch.nn as nn
# import torch


class Simple1DCNN(nn.Module):
    """
    Implements a simple 1D-CNN model for the Short-Time Temporal Regression Module
    in the Video Content Feature Processing Sub-Network.
    No need for max pooling here as the output must have the same length as the input.
    """
    def __init__(self, T):
        super(Simple1DCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ZeroPad1d((4, 0)),
            nn.Conv1d(in_channels=T, out_channels=T, kernel_size=5),
            nn.ReLU(), # TODO: validate if Relu is the correct activation function.
            # nn.MaxPool1d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.ZeroPad1d((4, 0)),
            nn.Conv1d(in_channels=T, out_channels=T, kernel_size=5),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential( # TODO: Does this layer make sense?
            nn.Flatten(0),
            nn.Linear(in_features=T*180, out_features=T*180),
            nn.ReLU(),
            nn.Unflatten(0, (T, 180))
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Group1DCNN(nn.Module): # TODO: Validate architecture. How input.shape = (T, 4) and output.shape = (T, 180)?
    """ 
    Implements a group 1D-CNN model for the Short-Time Temporal Regression Module
    in the QoS Feature Processing Sub-Network.
    No need for max pooling here as the output must have the same length as the input.
    """
    def __init__(self, T):
        super(Group1DCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ZeroPad1d((4, 0)),
            nn.Conv1d(in_channels=T, out_channels=T, groups=4, kernel_size=5),
            nn.ReLU() # TODO: validate if Relu is the correct activation function.
        )

        self.layer2 = nn.Sequential(
            nn.ZeroPad1d((4, 0)),
            nn.Conv1d(in_channels=T, out_channels=T, groups=4, kernel_size=5),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential( # TODO: Does this layer make sense?
            nn.Flatten(0),
            nn.Linear(in_features=T, out_features=T),
            nn.ReLU(),
            nn.Unflatten(0, (T, 180))
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x