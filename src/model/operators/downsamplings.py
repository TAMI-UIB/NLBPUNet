from torch import nn


class Downsamp_4_2(nn.Module):
    def __init__(self, channels, depthwise_multiplier):
        super(Downsamp_4_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels * depthwise_multiplier, kernel_size=5,
                               padding=5 // 2, groups=channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels * depthwise_multiplier, kernel_size=5,
                               padding=5 // 2, groups=channels)

    def forward(self, input):
        height = input.size(2)
        width = input.size(3)
        # Downsampling by a factor of 3
        x_d2 = self.conv2(input)[:, :, 0:height: 2, 0:width: 2]
        # Downsampling by a ratio of 2
        x_low = self.conv1(x_d2)[:, :, 0: int(height / 2): 2, 1: int(width / 2): 2]

        return x_d2, x_low
