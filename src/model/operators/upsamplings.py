from torch import nn
import torch

from .residual import EdgeProtector


class UpSamp_4_2(nn.Module):
    def __init__(self, in_channels, support_channels, kernel_size=3, depthwise_coef=1):
        super(UpSamp_4_2, self).__init__()
        self.conv2dTrans = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
        )
        self.conv2dTrans_factor2 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            stride=2,
            padding=5 // 2,
            output_padding=1,
        )
        self.edge_protector1 = EdgeProtector(in_channels, support_channels, kernel_size)
        self.edge_protector2 = EdgeProtector(in_channels, support_channels, kernel_size)
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * depthwise_coef,
            kernel_size=3,
            padding=1,
            groups=in_channels,
        )
        self.conv2d.weight.data = (1 / 16) * torch.ones(self.conv2d.weight.data.size())

    def forward(self, input, support_d2, support_full):

        input = self.conv2dTrans(input)

        input = self.edge_protector1(input, support_d2 / 10)

        input = self.conv2dTrans_factor2(input)
        input = self.edge_protector2(input, support_full / 10)

        input = self.conv2d(input)

        return input
