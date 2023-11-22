import os
import warnings

import pandas as pd
import rasterio as rio
import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from src.utils.transforms import tl_crop, center_crop

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

PYDEVD_THREAD_DUMP_ON_WARN_EVALUATION_TIMEOUT = 20000
PYDEVD_WARN_EVALUATION_TIMEOUT = 20000
pydev_do_not_trace = True


class FusionDatasetALSACE(torch.utils.data.Dataset):
    def __init__(self, dataset_path, new_crop_size, subset, **kwargs):
        super(FusionDatasetALSACE, self).__init__()

        self.downsamp_factor = kwargs.get('downsamp_factor', 32)
        self.crop_image = kwargs.get('crop_image', True)
        self.subset = subset
        self.new_crop_size = new_crop_size
        self.dataset_path = dataset_path
        self.to_tensor = ToTensor()
        self.index_df = self._get_index_df()
        self.total_patches = self.index_df['last_index'].max()-1
        self.images = [x[0] for x in os.walk(dataset_path) if os.path.isdir(x[0])][1:]
        self.hyper_channels = 3

    def __len__(self):
        val_size = self.total_patches // 3
        train_size = 2*val_size + self.total_patches % 3

        return train_size if self.subset == 'train' else val_size

    @staticmethod
    def get_in_channels():
        return 1

    def real_len(self):
        val_size = self.total_patches // 3
        train_size = 2 * val_size + self.total_patches % 3

        return train_size if self.subset == 'train' else val_size

    def __getitem__(self, index):
        # index = index // 4 if self.crop_image else index
        # crop_index = index % 4 if self.crop_image else None
        if self.subset == 'train':
            index = index + index//2 + 1
        else:
            index = index*3

        image_dict = self._get_image_path(index)
        gt = torch.load(f"{self.dataset_path}/{image_dict['path']}")
        patch_index = index - image_dict['first_index']
        gt = gt[patch_index][[2, 1, 0], :, :] / 1e4

        gt, pan, hs = self._generate_hs_and_pan(gt)
        # gt, pan, hs = self._generate_hs_and_pan(gt, crop_index)
        hs_lf = func.interpolate(hs, scale_factor=4)

        return gt, hs, pan, hs_lf

    def _generate_hs_and_pan(self, gt, index=None):
        downsamp_factor = self.downsamp_factor
        new_crop_size = self.new_crop_size
        crop_image = self.crop_image

        raw_gt_size = gt.size()

        if crop_image:
            # gt = indexed_crop(gt, (new_crop_size, new_crop_size), index)
            gt = center_crop(gt, (new_crop_size, new_crop_size))
        else:
            height = raw_gt_size[1] - (raw_gt_size[1] % downsamp_factor)
            width = raw_gt_size[2] - (raw_gt_size[2] % downsamp_factor)
            gt = tl_crop(gt, (height, width))

        size_gt = gt.size()

        hs = torch.zeros((size_gt[0], int(size_gt[1] / downsamp_factor), int(size_gt[2] / downsamp_factor)))
        for j in range(downsamp_factor):
            for k in range(downsamp_factor):
                hs = hs + gt[:, j:size_gt[1]:downsamp_factor, k:size_gt[2]:downsamp_factor] / (downsamp_factor**2)

        pan = torch.mean(gt, dim=0, keepdim=True)

        return gt, pan, hs

    def _get_image_path(self, index):
        aux = self.index_df[(self.index_df['first_index'] <= index) & (self.index_df['last_index'] > index)].to_dict(orient='records')
        return aux[0]

    def _get_index_df(self):
        df = pd.read_csv(f'{self.dataset_path}/index.csv', delimiter='\t')
        df = df[['tensor_05m_b2b3b4b8', 'nb_patches']]
        df = df.sort_values(by='nb_patches')
        df['first_index'] = df['nb_patches'].cumsum().shift(fill_value=0)
        df['last_index'] = df['nb_patches'].cumsum()
        df.rename(columns={"tensor_05m_b2b3b4b8": "path"}, inplace=True)
        return df
