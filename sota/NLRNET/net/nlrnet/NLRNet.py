# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
from torch.nn import Parameter
from .GCblock import ContextBlock2d

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

        self.gc = ContextBlock2d(64, 64)
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = self.gamma * output + identity_data
        output = self.gc(output)
        return output

class NLRNet(nn.Module):
    def __init__(self, dataset='gf2'):
        super(NLRNet, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv_input = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)
        #self.residual = self.make_layer(_Residual_Block, 16)
        self.residual = self.make_layer(_Residual_Block, 8)
        #self.residual = self.make_layer(_Residual_Block, 24)
        #self.residual = self.make_layer(_Residual_Block, 12)
        #self.residual = self.make_layer(_Residual_Block, 20)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)
        self.conv_output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1, stride=1, padding=0, bias=False),
        )

        # 无用模块
        self.outconv = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 5:
                    continue
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, *input):
        ms, pan = input[0], input[3]

        detail = torch.cat((ms, pan), dim=1)
        out = self.lrelu(self.conv_input(detail))

        residual = out
        out = self.residual(out)

        out = self.bn_mid(self.conv_mid(out))
        out = torch.mul(out,residual)
        out = self.conv_output(out)
        return out
class NLRNet_adapted(nn.Module):
    def __init__(self, dataset='gf2', **kwargs):
        super(NLRNet_adapted, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv_input = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)
        #self.residual = self.make_layer(_Residual_Block, 16)
        self.residual = self.make_layer(_Residual_Block, 8)
        #self.residual = self.make_layer(_Residual_Block, 24)
        #self.residual = self.make_layer(_Residual_Block, 12)
        #self.residual = self.make_layer(_Residual_Block, 20)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)
        self.conv_output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0, bias=False),
        )

        # 无用模块
        self.outconv = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 5:
                    continue
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, **kwargs):
        pan = kwargs['pan']
        ms = kwargs['hs']

        ms = torch.nn.functional.interpolate(ms, scale_factor=4, mode='bilinear')

        detail = torch.cat((ms, pan), dim=1)
        out = self.lrelu(self.conv_input(detail))

        residual = out
        out = self.residual(out)

        out = self.bn_mid(self.conv_mid(out))
        out = torch.mul(out, residual)
        out = self.conv_output(out)
        return dict(fused=out)
