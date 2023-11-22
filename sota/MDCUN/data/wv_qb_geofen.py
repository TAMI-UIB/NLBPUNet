
# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
import h5py
import numpy as np
import torch
import torch.nn.functional as func
import torch.utils.data as data


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path, img_scale):
        super(Dataset_Pro, self).__init__()

        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale

        self.ms = torch.from_numpy(ms1)

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:

        if 'valid' in file_path:
            self.gt = self.gt.permute([0, 2, 3, 1])

    def __getitem__(self, index):
        gt = self.gt[index, :, :, :].float()
        pan = self.pan[index, :, :, :].float()
        hs = self.ms[index, :, :, :].float()
        return gt, pan, hs

    def __len__(self):
        return self.gt.shape[0]


class WV3_QB_Geofen(data.Dataset):
    def __init__(self, dataset_path, new_crop_size, subset, **kwargs):
        super(WV3_QB_Geofen, self).__init__()
        self.downsamp_factor = kwargs.get('downsamp_factor', 32)
        self.crop_image = kwargs.get('crop_image', False)
        self.subset = subset
        self.dataset_path = dataset_path

        if self.crop_image == True:
            print("This dataset only allows cro_image=False.")
            exit(1)
        if self.downsamp_factor != 4:
            print("This dataset only allows downsampling factor 4")
            exit(2)

        data = h5py.File(f'{dataset_path}/{subset}/data.h5')  # NxCxHxW = 0x1x2x3
        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        img_scale = self.get_img_scale(dataset_path)
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:
        self.hyper_channels = self.gt.shape[1]
        self.multi_channels = 1

    @staticmethod
    def get_in_channels():
        return 1

    def __getitem__(self, index):
        gt = self.gt[index, :, :, :].float()
        pan, hs = self._generate_hs_and_pan(gt)
        hs_lf = func.interpolate(hs, scale_factor=4)
        return gt, hs, pan, hs_lf

    def __len__(self):
        return self.gt.shape[0]

    def real_len(self):
        return self.gt.shape[0]

    def get_img_scale(self, dataset_path):
        if "WorldView3" in dataset_path:
            return 2**11-1
        if "Gaofen" in dataset_path:
            return 2**10-1
        if "QuickBird" in dataset_path:
            return 2**11-1

    def _generate_hs_and_pan(self, gt):
        downsamp_factor = self.downsamp_factor

        size_gt = gt.size()

        hs = torch.zeros((size_gt[0], int(size_gt[1] / downsamp_factor), int(size_gt[2] / downsamp_factor)))
        for j in range(downsamp_factor):
            for k in range(downsamp_factor):
                hs = hs + gt[:, j:size_gt[1]:downsamp_factor, k:size_gt[2]:downsamp_factor] / (downsamp_factor**2)

        pan = torch.mean(gt, dim=0, keepdim=True)
        return pan, hs
