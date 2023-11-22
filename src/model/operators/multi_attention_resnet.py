import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self,  kernel_size, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        features = self.conv1(x)
        features = self.relu(features)
        features = self.conv2(features)
        return self.relu(features + x)


class MultiHeadAttentionEsDot(nn.Module):
    def __init__(self, u_channels, pan_channels, patch_size, window_size):
        super(MultiHeadAttentionEsDot, self).__init__()
        self.geometric_head = SelfAttentionEsDot(u_channels=u_channels, pan_channels=pan_channels, patch_size=patch_size, window_size=window_size)
        self.spectral_head = SelfAttentionEsDot(u_channels=u_channels, pan_channels=u_channels, patch_size=1, window_size=window_size)
        self.mix_head = SelfAttentionEsDot(u_channels=u_channels, pan_channels=pan_channels+u_channels, patch_size=patch_size, window_size=window_size)
        self.mlp = nn.Linear(3, 1)

    def forward(self, u, pan):
        head1 = self.geometric_head(u, pan)
        head2 = self.spectral_head(u, u)

        head3 = self.mix_head(u, torch.concat([u, pan], dim=1))

        return self.mlp(torch.concat([head1, head2, head3], dim=4)).squeeze(4)


class SelfAttentionEsDot(nn.Module):
    def __init__(self, u_channels, pan_channels, patch_size, window_size):
        super(SelfAttentionEsDot, self).__init__()
        self.pan_channels = pan_channels
        self.u_channels = u_channels
        self.patch_size = patch_size
        self.window_size = window_size
        self.spatial_weights = SpatialWeightsEsDot(channels=pan_channels, window_size=window_size, patch_size=patch_size)
        self.g = nn.Conv2d(u_channels, u_channels, 1, bias=False)

    def forward(self, u, pan):
        b, c, h, w = u.size()
        weights = self.spatial_weights(pan)
        g = self.g(u)  # [b, 3, h, w]
        g = F.unfold(g, self.window_size, padding=self.window_size // 2)
        g = g.view(b, self.u_channels, self.window_size * self.window_size, -1)
        g = g.view(b, self.u_channels, self.window_size * self.window_size, h, w)
        g = g.permute(0, 3, 4, 2, 1)
        return torch.matmul(weights, g).permute(0, 4, 1, 2, 3)


class SpatialWeightsEsDot(torch.nn.Module):
    def __init__(self, channels,  window_size, patch_size):
        super(SpatialWeightsEsDot, self).__init__()
        self.channels = channels
        self.phi = nn.Conv2d(channels, channels, 1, bias=False)
        self.theta = nn.Conv2d(channels, channels, 1, bias=False)
        self.window_size = window_size
        self.patch_size = patch_size
        self.softmax = nn.Softmax(dim=-1)
        self.eps = 1e-6

    def forward(self, u):
        b, c, h, w = u.size()
        phi = self.phi(u)
        theta = self.phi(u)
        theta = F.unfold(theta, self.patch_size, padding=self.patch_size // 2)
        theta = theta.view(b, 1, c*self.patch_size * self.patch_size, -1)
        theta = theta.view(b, 1, c*self.patch_size * self.patch_size, h, w)
        theta = theta.permute(0, 3, 4, 1, 2)

        phi = F.unfold(phi, self.patch_size, padding=self.patch_size // 2)
        phi = phi.view(b, c * self.patch_size * self.patch_size, h, w)
        phi = F.unfold(phi, self.window_size, padding=self.window_size // 2)
        phi = phi.view(b, c * self.patch_size * self.patch_size, self.window_size * self.window_size, h, w)
        phi = phi.permute(0, 3, 4, 1, 2)

        att = torch.matmul(theta, phi)

        return self.softmax(att)
