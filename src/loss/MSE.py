from torch import nn


class MSE(nn.Module):
    def __init__(self, **kwargs):
        super(MSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, gt, fused, **kwargs):
        mse = self.mse(gt, fused)
        return mse, {}