import torch.nn as nn


class FC1(nn.Module):
    """
    Implementation of the 1st fully connected layer for the dual-stage attention model.
    Ref.: https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
    """
    def __init__(self):
        super(FC1, self).__init__()
        self.layer1 = nn.Linear(7936, 180)

    def forward(self, x):
        return self.layer1(x)


class FC2(nn.Module):
    """
    Implementation of the 2nd fully connected layer for the dual-stage attention model.
    Ref.: https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
    """
    def __init__(self):
        super(FC2, self).__init__()
        self.layer1 = nn.Linear(7936, 180)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(180, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)

        return x


class FC3(nn.Module):
    """
    Implementation of the 3rd fully connected layer for the dual-stage attention model.
    Ref.: https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
    """
    def __init__(self):
        super(FC3, self).__init__()
        self.layer1 = nn.Linear(360, 180)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(180, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)

        return x


class FC4(nn.Module):
    """
    Implementation of the 4th fully connected layer for the dual-stage attention model.
    Ref.: https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
    """
    def __init__(self):
        super(FC4, self).__init__()
        self.layer1 = nn.Linear(7936, 180)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(180, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)

        return x


class FC5(nn.Module):
    """
    Implementation of the 5th fully connected layer for the dual-stage attention model.
    Ref.: https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
    """
    def __init__(self):
        super(FC5, self).__init__()
        self.layer1 = nn.Linear(360, 180)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(180, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)

        return x