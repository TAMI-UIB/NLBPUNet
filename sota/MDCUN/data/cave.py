import os
import warnings

import rasterio as rio
import torch
from PIL import Image
import torch.nn.functional as func
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from src.utils.transforms import indexed_crop, hiperspectral_to_rgb, tl_crop

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

PYDEVD_THREAD_DUMP_ON_WARN_EVALUATION_TIMEOUT = 20000
PYDEVD_WARN_EVALUATION_TIMEOUT = 20000
pydev_do_not_trace = True


class FusionDatasetCAVE(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        new_crop_size,
        subset='train',
        multi_spectral=False,
        crop_image=True,
        downsamp_factor=32
    ):
        super(FusionDatasetCAVE, self).__init__()

        self.downsamp_factor = downsamp_factor
        self.multi_spectral = multi_spectral
        self.crop_image = crop_image
        self.new_crop_size = new_crop_size
        self.to_tensor = ToTensor()
        self.images = [x[0] for x in os.walk(f'{dataset_path}/{subset}') if os.path.isdir(x[0])][1:]
        self.hyper_channels = 31

    def __len__(self):
        return len(self.images)*16 if self.crop_image else len(self.images)

    def get_in_channels(self):
        return 3 if self.multi_spectral else 1

    def __getitem__(self, index):

        image = index // 16 if self.crop_image else index
        crop_index = index % 16 if self.crop_image else None

        image_file = sorted(os.listdir(self.images[image]))
        image_file = [im for im in image_file if ".png" in im]
        w, h = Image.open(os.path.join(self.images[image], image_file[0])).size
        gt = torch.zeros(len(image_file), h, w)
        for channel, file in enumerate(image_file):
            file_path = os.path.join(self.images[image], file)
            im = self.to_tensor(Image.open(file_path))
            gt[channel] = torch.div(im[0], 2 ** 16 - 1)

        # generate new hs and pan images from the original hs (considered as ground truth)
        gt, pan, hs = self._generate_hs_and_pan(gt, crop_index)
        hs_lf = func.interpolate(hs, scale_factor=4)
        return gt, hs, pan, hs_lf

    def _generate_hs_and_pan(self, gt, index=None):
        downsamp_factor = self.downsamp_factor
        new_crop_size = self.new_crop_size
        crop_image = self.crop_image

        raw_gt_size = gt.size()

        if crop_image:
            gt = indexed_crop(gt, (new_crop_size, new_crop_size), index)
        else:
            height = raw_gt_size[1] - (raw_gt_size[1] % downsamp_factor)
            width = raw_gt_size[2] - (raw_gt_size[2] % downsamp_factor)
            gt = tl_crop(gt, (height, width))


        size_gt = gt.size()

        hs = torch.zeros((size_gt[0], int(size_gt[1] / downsamp_factor), int(size_gt[2] / downsamp_factor)))
        for j in range(downsamp_factor):
            for k in range(downsamp_factor):
                hs = hs + gt[:, j:size_gt[1]:downsamp_factor, k:size_gt[2]:downsamp_factor] / (downsamp_factor**2)

        rgb = hiperspectral_to_rgb(gt)
        pan = torch.mean(gt, dim=0, keepdim=True)

        pan = rgb if self.multi_spectral else pan

        return gt, pan, hs
