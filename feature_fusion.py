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
        O_VC = x[0]  # Video content features
        O_QOS = x[1]  # QoS features
        X_VC = mean(O_VC, dim=0, keepdim=True)
        X_QOS = mean(O_QOS, dim=0, keepdim=True)

        overall_alpha = self.FC2(cat((X_VC, X_QOS), dim=1))
        continuous_alpha = self.FC4(cat((O_VC, O_QOS), dim=1))

        overall_QoE = self.FC3(cat(overall_alpha*X_VC, (1 - overall_alpha)*X_QOS))
        continuous_QoE = self.FC5(cat(continuous_alpha*O_VC, (1 - continuous_alpha)*O_QOS))

        return overall_QoE, continuous_QoE