from os import environ
from os.path import join

import h5py
import torch
import torch.utils.data as data


class StMichel(data.Dataset):
    def __init__(self, fold):
        super(StMichel, self).__init__()
        self.fold = fold
        dataset_path = join(environ['DATASET_PATH'], "stmichel")

        print(f'######### {dataset_path}/data.h5 #########')
        data = h5py.File(f'{dataset_path}/data.h5')  # NxCxHxW = 0x1x2x3
        # tensor type:
        if fold not in ['train', 'validation']:
            print("This dataset only allows subset=train or subset=validation")
            exit(3)

        self.gt = data[fold]['gt']
        self.pan = data[fold]['pan']
        self.hs = data[fold]['low']
        self.hyper_channels = self.gt[0].shape[0]
        self.multi_channels = self.pan[0].shape[0]

    def __getitem__(self, index):
        gt = torch.tensor(self.gt[index, :, :, :], dtype=torch.float32)
        pan = torch.tensor(self.pan[index, :, :, :], dtype=torch.float32)
        hs = torch.tensor(self.hs[index, :, :, :], dtype=torch.float32)
        return dict(gt=gt, pan=pan, hs=hs)

    def __len__(self):
        return self.gt.shape[0]
