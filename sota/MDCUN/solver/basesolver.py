#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:07:03
LastEditTime: 2020-11-25 19:24:54
@Description: file content
'''
import datetime
import os
import time

import torch
from torch.utils.data import DataLoader

from sota.MDCUN.data.data import *
from sota.MDCUN.data import dict_dataset

class BaseSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nEpochs = cfg['nEpochs']
        self.checkpoint_dir = cfg['checkpoint']
        self.epoch = 1

        self.timestamp = int(time.time())
        self.now_time = datetime.datetime.now().strftime("%m-%d-%H-%M")
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_ref_results = {'metrics: ': '  PSNR,     SSIM,   ... SCC,    Q', 'deep   ': [0, 0, 0, 0, 0, 0]}
        self.best_no_ref_results = {'metrics: ': '  D_lamda,  D_s,    QNR', 'deep    ': [0, 0, 0]}
        self.downsamp_factor = cfg['data']['upscale']

        self.dataset = dict_dataset[cfg['dataset']]
        self.dataset_path = os.path.join(cfg['data_dir_train'], cfg['dataset'])

        if cfg['gpu_mode']:
            self.num_workers = cfg['threads']
        else:
            self.num_workers = 0

        self.train_dataset = self.dataset(self.dataset_path, cfg['data']['patch_size'], 'train',
                                          downsamp_factor=self.downsamp_factor, multi_spectral=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True,
                                       num_workers=self.num_workers)

        self.val_dataset = self.dataset(self.dataset_path, cfg['data']['patch_size'], 'validation',
                                        downsamp_factor=self.downsamp_factor, multi_spectral=False, crop_image=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False,
                                     num_workers=self.num_workers)

        self.records = {'Epoch': [], 'PSNR': [], 'SSIM': [], 'Loss': []}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def load_checkpoint(self, model_path):
        if os.path.exists(model_path):
            ckpt = torch.load(model_path)
            self.epoch = ckpt['epoch']
            self.records = ckpt['records']
        else:
            raise FileNotFoundError

    def save_checkpoint(self):
        self.ckp = {
            'epoch': self.epoch,
            'records': self.records,
        }

    def train(self):
        raise NotImplementedError
    
    def eval(self):
        raise NotImplementedError
    
    def run(self):
        while self.epoch <= self.nEpochs:
            self.train()
            self.eval()
            self.save_checkpoint()
            # self.save_records()
            self.epoch += 1
        #self.logger.log('Training done.')
