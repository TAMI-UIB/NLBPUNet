import torch
from torch import nn


class EdgeProtector(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size, feature=3):
        super(EdgeProtector, self).__init__()
        init_channels = x_channels + y_channels
        mid_channels = x_channels + y_channels + feature
        final_channels = x_channels
        self.convbatchnorm1 = ConvBatchnormRelu(in_channels=init_channels, out_channels=mid_channels,
                                                kernel_size=kernel_size)
        self.convbatchnorm2 = ConvBatchnormRelu(in_channels=mid_channels, out_channels=mid_channels,
                                                kernel_size=kernel_size)
        self.convbatchnorm3 = ConvBatchnormRelu(in_channels=mid_channels, out_channels=final_channels,
                                                kernel_size=kernel_size)

    def forward(self, x, y):
        features_1 = self.convbatchnorm1(torch.cat((x, y), dim=1))
        features_2 = self.convbatchnorm2(features_1)
        features_3 = self.convbatchnorm3(features_2)
        features_3 = features_3 + x
        return features_3


class ConvBatchnormRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBatchnormRelu, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                                  nn.BatchNorm2d(out_channels, eps=1e-05), nn.ReLU(inplace=True))
        torch.nn.init.zeros_(self.conv[0].bias)

    def forward(self, x):
        x = self.conv(x)
        return x
