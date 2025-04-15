import torch
import torch.nn as nn

from torch.nn import MSELoss


class Loss(nn.Module):
    """
    Custom loss function for the model.
    Loss = MSE(ŷ, ylabel) + 0.1 * (1 - PLCC(ŷ, ylabel))
    Implemented following suggestion @ https://discuss.pytorch.org/t/how-to-implement-a-custom-loss-in-pytorch/197938/3.
    """
    def __init__(self):
        super(Loss, self).__init__()
        self.mse = MSELoss()
        self.PLCC = self.pearson_loss
        self.alpha = 0.1
        self.eps = 1e-8

    def forward(self, y, y_hat):
        loss = self.mse(y[0], y_hat[0]) + self.alpha * (1 - self.PLCC(y[0], y_hat[0])) + \
        self.mse(y[1], y_hat[1]) + self.alpha * (1 - self.PLCC(y[1], y_hat[1]))

        return loss

    def pearson_loss(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        stacked = torch.stack([pred_flat, target_flat], dim=0)
        
        corr_matrix = torch.corrcoef(stacked)
        
        plcc = corr_matrix[0, 1]
        
        plcc = torch.clamp(plcc, -1.0 + self.eps, 1.0 - self.eps)
        
        return plcc