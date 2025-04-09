import torch.nn as nn

from torch.nn import MSELoss
from scipy.stats import pearsonr


class Loss(nn.Module):
    """
    Custom loss function for the model.
    Loss = MSE(ŷ, ylabel) + 0.1 * (1 - PLCC(ŷ, ylabel))
    Implemented following suggestion @ https://discuss.pytorch.org/t/how-to-implement-a-custom-loss-in-pytorch/197938/3.
    """
    def __init__(self):
        super(Loss, self).__init__()
        self.mse = MSELoss()
        self.PLCC = lambda x, y: pearsonr(x, y)
        self.alpha = 0.1

    def forward(self, y, y_hat):
        return self.mse(y, y_hat) + self.alpha * (1 - self.PLCC(y, y_hat))