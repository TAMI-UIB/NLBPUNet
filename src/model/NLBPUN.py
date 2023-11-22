import torch.nn

import torch
from torch import nn

from .operators.downsamplings import Downsamp_4_2
from .operators.multi_attention_resnet import ResBlock, MultiHeadAttentionEsDot

from .operators.upsamplings import UpSamp_4_2

class MARNetNZ(torch.nn.Module):
    def __init__(self, u_channels, pan_channels, features_channels, patch_size, window_size, kernel_size=3):
        super().__init__()
        self.N, self.C, self.H, self.W = 1, 3, 256, 256

        self.features_channels = features_channels
        # Residual blocks and related convolution
        self.ResNet_features = nn.Conv2d(in_channels=u_channels, out_channels=features_channels,
                                         kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size // 2)
        self.res1 = ResBlock(kernel_size=kernel_size, in_channels=features_channels+5+5)
        self.res2 = ResBlock(kernel_size=kernel_size, in_channels=features_channels+5+5)
        self.res3 = ResBlock(kernel_size=kernel_size, in_channels=features_channels+5+5)
        self.ResNet_recon = nn.Conv2d(in_channels=features_channels+5+5, out_channels=u_channels, kernel_size=kernel_size,
                                      stride=1, bias=False, padding=kernel_size // 2)
        # Multi Attention blocks and related convolution
        self.MultiAtt_features_u = nn.Conv2d(in_channels=u_channels, out_channels=5, kernel_size=kernel_size, stride=1,
                                             bias=False, padding=kernel_size // 2)
        self.ResNet_features_inter = nn.Conv2d(in_channels=u_channels, out_channels=5, kernel_size=kernel_size, stride=1,
                                               bias=False, padding=kernel_size // 2)
        self.MultiAtt_features_pan = nn.Conv2d(in_channels=pan_channels, out_channels=3, kernel_size=kernel_size, stride=1,
                                               bias=False, padding=kernel_size // 2)

        self.MultiAtt_recon = nn.Conv2d(in_channels=5, out_channels=u_channels, kernel_size=kernel_size, stride=1,
                                        bias=False, padding=kernel_size // 2)
        self.multi_head = MultiHeadAttentionEsDot(u_channels=5, pan_channels=3, patch_size=patch_size,
                                                  window_size=window_size)

    def forward(self, u, inter, pan):
        # Multi Attention Component
        u_features = self.MultiAtt_features_u(u)
        pan_features = self.MultiAtt_features_pan(pan)
        u_multi_att = self.multi_head(u_features, pan_features)
        # Residual Component
        inter_features = self.ResNet_features_inter(inter)
        u_features = self.ResNet_features(u)
        u_aux = torch.cat([u_multi_att, u_features, inter_features], dim=1)
        res1 = self.res1(u_aux)
        res2 = self.res2(res1)
        res3 = self.res3(res2)

        return self.ResNet_recon(res3)


class NLBP(nn.Module):
    def __init__(self, hyper_channels, multi_channels):
        super().__init__()
        self.MARNet = MARNetNZ(u_channels=hyper_channels, pan_channels=multi_channels,
                               features_channels=32, patch_size=3, window_size=7, kernel_size=3)

    def forward(self, u, pan_lf, pan, hs_inter):
        return u+self.MARNet(hs_inter*pan-u*pan_lf, hs_inter, pan)


class NLBPUN(nn.Module):
    def __init__(
            self,
            hyper_channels,
            multi_channels,
            iter_stages,
            device='cpu',
            std_noise=None,
            **kwargs
    ):
        super(NLBPUN, self).__init__()
        self.sampling = kwargs['downsamp_factor']
        self.downsamp_ms = Downsamp_4_2(channels=multi_channels, depthwise_multiplier=1)
        self.upsamp_lf = nn.Sequential(*[nn.ConvTranspose2d(hyper_channels, hyper_channels, stride=4, kernel_size=3,
                                                            padding=1, output_padding=3, bias=False),
                                         nn.Conv2d(hyper_channels, hyper_channels, kernel_size=3, padding=1, bias=False)])
        self.upsamp_ms = nn.Sequential(*[nn.ConvTranspose2d(multi_channels, multi_channels, stride=4, kernel_size=3,
                                                            padding=1, output_padding=3, bias=False),
            nn.Conv2d(multi_channels, multi_channels, kernel_size=3, padding=1, bias=False)])
        self.NZDBP = nn.ModuleList([
                NLBP(hyper_channels=hyper_channels, multi_channels=multi_channels)
                for i in range(iter_stages+1)
            ])
        self.upsamp = UpSamp_4_2(hyper_channels, multi_channels)
        self.iter_stages = iter_stages
        self.device = device

    def forward(self, pan, hs):
        pan_d2, pan_low = self.downsamp_ms(pan)
        pan_lf = self.upsamp_ms(pan_low)
        hf_up = self.upsamp_lf(hs)

        u = self.upsamp(hs, pan_d2, pan)

        for i in range(self.iter_stages):
            u = self.NZDBP[i](u, pan_lf, pan, hf_up)

        return u, None, self.DB(u)-hs, None

    def DB(self, u):
        size = u.size()
        DBu = torch.zeros((size[0],size[1], int(size[2] / self.sampling), int(size[3] / self.sampling))).to(self.device)
        for j in range(self.sampling):
            for k in range(self.sampling):
                DBu = DBu + u[:,:, j:size[2]:self.sampling, k:size[3]:self.sampling] / (self.sampling ** 2)
        return DBu
