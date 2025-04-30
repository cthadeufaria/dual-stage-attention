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

    def forward(self, y_hat, y):
        continuous_labels = [a[1] for a in y]
        continuous_predictions = [a[1].squeeze(-1) for a in y_hat]

        continuous_loss = 0.
        for prediction, label in zip(continuous_predictions, continuous_labels):
            continuous_loss += self.mse(prediction, label) + self.alpha * (1. - self.PLCC(prediction, label))

        continuous_loss /= len(continuous_labels)

        overall_labels = torch.stack([a[0].float() for a in y])
        overall_predictions = torch.stack([a[0] for a in y_hat]).squeeze(-1)
        overall_loss = self.mse(overall_predictions, overall_labels) + self.alpha * (1. - self.PLCC(overall_predictions, overall_labels))

        if torch.isnan(overall_loss).any():
            loss = continuous_loss

        else:
            loss = continuous_loss + overall_loss
            loss /= 2.            

        return loss

    def pearson_loss(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        covariance = torch.cov(torch.stack([pred, target]))[0, 1]
        std_pred = torch.std(pred)
        std_target = torch.std(target)

        plcc = covariance / (std_pred * std_target + self.eps)
        plcc = torch.clamp(plcc, -1.0 + self.eps, 1.0 - self.eps)
        
        return plcc