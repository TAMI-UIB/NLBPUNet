import warnings

import h5py
import numpy as np
import rasterio as rio
import torch
import torch.nn.functional as func
import torch.utils.data as data
from torch.utils.data import DataLoader

from src.utils.transforms import add_noise

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

PYDEVD_THREAD_DUMP_ON_WARN_EVALUATION_TIMEOUT = 20000
PYDEVD_WARN_EVALUATION_TIMEOUT = 20000
pydev_do_not_trace = True


class QuickBirdTest(data.Dataset):
    def __init__(self, dataset_path, new_crop_size, subset, **kwargs):
        super(QuickBirdTest, self).__init__()
        self.downsamp_factor = kwargs.get('downsamp_factor', 32)
        self.crop_image = kwargs.get('crop_image', False)
        self.subset = "test"
        self.dataset_path = dataset_path
        self.patch_size = kwargs.get('patch_size')

        if self.crop_image == True:
            print("This dataset only allows crop_image=False.")
            exit(1)
        if self.downsamp_factor != 4:
            print("This dataset only allows downsampling factor 4")
            exit(2)
        print(f'######### {dataset_path}/{self.subset}/data.h5 #########')
        data = h5py.File(f'{dataset_path}/{self.subset}/data.h5')  # NxCxHxW = 0x1x2x3
        # tensor type:
        self.gt = torch.from_numpy(np.array(data['gt'], dtype=np.float32))/2048.
        self.hyper_channels = self.gt.shape[1]
        self.multi_channels = 1

    @staticmethod
    def get_in_channels():
        return 1
    def get_len(self):
        return self.gt.shape[0]
    def __getitem__(self, index):
        gt = self.gt[index, :, :, :].float()
        pan, hs = self._generate_hs_and_pan(gt)
        hs_lf = func.interpolate(hs, scale_factor=4)

        # hs = add_noise(hs, 10/255, gt.device)
        return gt, hs, pan, hs_lf

    def __len__(self):
        return self.gt.shape[0]
        # return 20

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


if __name__ == '__main__':
    dataset = QuickBirdTest
    dataset_path = "/home/user/Recerca/UIB/datasets/QuickBirdTest"
    dataset = dataset(dataset_path, 64, 'test', downsamp_factor=4, crop_image=False, patch_size=64)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    for i, d in enumerate(data_loader):
        print(d[0].size())