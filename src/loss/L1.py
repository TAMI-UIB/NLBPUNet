from torch import nn


class L1(nn.Module):
    def __init__(self, **kwargs):
        super(L1, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, gt, fused, **kwargs):
        l1 = self.l1(gt, fused)
        return l1, {}
