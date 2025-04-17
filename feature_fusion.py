from torch import mean, cat
import torch.nn as nn
from fully_connected_networks import FC2, FC3, FC4, FC5


class FeatureFusion(nn.Module):
    """
    Implements the feature fusion module for the dual-stage attention model.
    """
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.FC2 = FC2()
        self.FC3 = FC3()
        self.FC4 = FC4()
        self.FC5 = FC5()

    def forward(self, x):
        O_vc = x[0]  # Video Content
        O_QoS = x[1]  # QoS
        X_vc = mean(O_vc, dim=0)
        X_QoS = mean(O_QoS, dim=0)

        overall_alpha = self.FC2(cat((X_vc, X_QoS)))
        continuous_alpha = self.FC4(cat((O_vc, O_QoS), dim=1))

        overall_QoE = self.FC3(cat((overall_alpha*X_vc, (1 - overall_alpha)*X_QoS)))
        continuous_QoE = self.FC5(cat((continuous_alpha*O_vc, (1 - continuous_alpha)*O_QoS), dim=1))

        return [overall_QoE, continuous_QoE]