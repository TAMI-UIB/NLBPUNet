#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:04:48
LastEditTime: 2020-12-03 22:02:20
@Description: file content
'''
import importlib
import os
import shutil
# from performance.f_cal import cal_performance
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm

from sota.MDCUN.model.custom_pan_unfolding_v4 import pan_unfolding
from sota.MDCUN.solver.basesolver import BaseSolver
from sota.MDCUN.utils.config import save_yml
from sota.MDCUN.utils.utils import maek_optimizer, make_loss, save_config, \
    save_net_config
from src.utils.metrics import MetricCalculator
from src.utils.transforms import hiperspectral_to_rgb

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Solver(BaseSolver):
    def __init__(self, cfg, cuda=False):
        super(Solver, self).__init__(cfg)
        self.init_epoch = self.cfg['schedule']
        
        # net_name = self.cfg['algorithm'].lower()
        # lib = importlib.import_module('model.' + net_name)
        # # net = lib.Net
        # net = lib.pan_unfolding
        net = pan_unfolding
        self.cuda = cuda
        self.model = net(
            hyper_channels=self.cfg['in_channels'],
            T=self.cfg['stage']
        )

        self.optimizer = maek_optimizer(self.cfg['schedule']['optimizer'], cfg, self.model.parameters())
        self.loss = make_loss(self.cfg['schedule']['loss'])
        #self.vggloss = make_loss('VGG54')
        now = datetime.now()
        time = now.strftime("%Y-%m-%d:%H:%M:%S")
        self.log_name = f"mdcun_{time}"
        # save log
        self.log_dir = f"{os.environ['SNAPSHOT_PATH']}/{self.cfg['dataset']}/sota/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir + str(self.log_name))

        save_net_config(self.log_dir+self.log_name, self.model)  # TODO: Change with our time

        save_yml(cfg, os.path.join(self.log_dir+self.log_name, 'config.yml'))
        save_config(self.log_dir+self.log_name, 'Train dataset has {} images and {} batches.'.format(len(self.train_dataset), len(self.train_loader)))
        save_config(self.log_dir+self.log_name, 'Val dataset has {} images and {} batches.'.format(len(self.val_dataset), len(self.val_loader)))
        save_config(self.log_dir+self.log_name, 'Model parameters: ' + str(sum(param.numel() for param in self.model.parameters())))

    def train(self): 
        with tqdm(total=len(self.train_loader), miniters=1,
                  desc='Initial Training Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t:

            epoch_loss = 0
            for iteration, batch in enumerate(self.train_loader, 1):
                ms_image, lms_image, pan_image, bms_image = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])

                if self.cuda:
                    ms_image, lms_image, pan_image, bms_image = ms_image.cuda(self.gpu_ids[0]), lms_image.cuda(self.gpu_ids[0]), pan_image.cuda(self.gpu_ids[0]), bms_image.cuda(self.gpu_ids[0])
                self.optimizer.zero_grad()
                self.model.train()

                y = self.model(lms_image, bms_image, pan_image)
                loss = self.loss(y, ms_image) / (self.cfg['data']['batch_size'] * 2)
                # y, u, v = self.model(lms_image, bms_image, pan_image)
                # loss = (self.loss(y, ms_image) / (self.cfg['data']['batch_size'] * 2) + self.loss(u, ms_image) / (self.cfg['data']['batch_size'] * 2) + self.loss(v, ms_image) / (self.cfg['data']['batch_size'] * 2))/3

                ## TODO: THIS CODE IS COMENTED BY US!

                # if self.cfg['schedule']['use_YCbCr']:
                #     y_vgg = torch.unsqueeze(y[:,3,:,:], 1)
                #     y_vgg_3 = torch.cat([y_vgg, y_vgg, y_vgg], 1)
                #     pan_image_3 = torch.cat([pan_image, pan_image, pan_image], 1)
                #     vgg_loss = self.vggloss(y_vgg_3, pan_image_3)

                epoch_loss += loss.data
                #epoch_loss = epoch_loss + loss.data + vgg_loss.data
                t.set_postfix_str("Batch loss {:.4f}".format(loss.item()))
                t.update()

                loss.backward()
                # print("grad before clip:"+str(self.model.output_conv.conv.weight.grad))
                if self.cfg['schedule']['gclip'] > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg['schedule']['gclip']
                    )
                self.optimizer.step()
                
            self.records['Loss'].append(epoch_loss / len(self.train_loader))
            # self._add_image('train MS', ms_image[0], self.epoch)
            # self._add_image('train Fused', y[0], self.epoch)
            # self._add_image('train Pan', pan_image[0], self.epoch)
            save_config(self.log_dir+self.log_name, 'Initial Training Epoch {}: Loss={:.4f}'.format(self.epoch, self.records['Loss'][-1]))
            self.writer.add_scalar('Loss_epoch', self.records['Loss'][-1], self.epoch)

    def eval(self):
        with tqdm(total=len(self.val_loader), miniters=1,
                desc='Val Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t1:
            psnr_list, ssim_list = [], []

            metrics = MetricCalculator(self.val_dataset.real_len())
            for iteration, batch in enumerate(self.val_loader, 1):
                ms_image, lms_image, pan_image, bms_image = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
                if self.cuda:
                    ms_image, lms_image, pan_image, bms_image = ms_image.cuda(0), lms_image.cuda(0), pan_image.cuda(0), bms_image.cuda(0)

                self.model.eval()
                with torch.no_grad():
                    y = self.model(lms_image, bms_image, pan_image)
                    # y, u, v  = self.model(lms_image, bms_image, pan_image)
                    loss = self.loss(y, ms_image)
                metrics.update(y.cpu(), ms_image.cpu())

                t1.set_postfix_str(f'Batch loss: {loss.item()}')
                t1.update()
            avg_psnr = metrics.dict['psnr']
            avg_ssim = metrics.dict['ssim']
            print(avg_psnr)
            print(metrics.dict)

            if avg_psnr > self.best_psnr:
                ckpt = {"model": self.model, "state_dict": self.model.state_dict}
                torch.save(ckpt, os.environ['PROJECT_PATH'] + "/sota/MDCUN/log/weights_best.pth")
                self.writer.add_text("best metrics:", str(metrics.dict))

            ckpt = {"model": self.model, "state_dict": self.model.state_dict}
            torch.save(ckpt, os.environ['PROJECT_PATH'] + "/sota/MDCUN/log/weights_last.pth")

            self.best_psnr = max(self.best_psnr, avg_psnr)
            self.best_ssim = max(self.best_ssim, avg_ssim)

            t1.set_postfix_str('Batch loss: {:.4f}, PSNR: {:.4f}/{:.4f}, SSIM: {:.4f}/{:.4f}'.format(loss.item(), avg_psnr, self.best_psnr, avg_ssim, self.best_ssim))
            t1.update()

            self.records['Epoch'].append(self.epoch)
            self.records['PSNR'].append(metrics.dict['psnr'])
            self.records['SSIM'].append(metrics.dict['ssim'])

            save_config(self.log_dir+self.log_name, 'Val Epoch {}: PSNR={:.4f}, SSIM={:.4f}'.format(self.epoch, self.records['PSNR'][-1],
                                                                 self.records['SSIM'][-1]))
            self.writer.add_histogram('Fused', y, global_step=self.epoch)
            self.writer.add_histogram('GT', ms_image, global_step=self.epoch)
            self.writer.add_scalar('PSNR_epoch', self.records['PSNR'][-1], global_step=self.epoch)
            self.writer.add_scalar('SSIM_epoch', self.records['SSIM'][-1], global_step=self.epoch)
            self._add_image('val MS (gt)', ms_image[0], self.epoch)
            self._add_image('val Fused', y[0], self.epoch)
            self._add_image('val Pan', pan_image[0], self.epoch)

    def test(self):
        print("test phase")
        self.model.eval()
        avg_time = []
        for batch in self.data_loader:
            ms_image, lms_image, pan_image, bms_image, name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), (batch[4])
            if self.cuda:
                ms_image = ms_image.cuda(0)
                lms_image = lms_image.cuda(0)
                pan_image = pan_image.cuda(0)
                bms_image = bms_image.cuda(0)

            t0 = time.time()
            with torch.no_grad():
                prediction = self.model(lms_image, bms_image, pan_image)
            t1 = time.time()

            if self.cfg['data']['normalize']:
                ms_image = (ms_image+1) /2
                lms_image = (lms_image+1) /2
                pan_image = (pan_image+1) /2
                bms_image = (bms_image+1) /2

            # print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            avg_time.append(t1 - t0)
            self.save_img(bms_image.cpu().data, name[0][0:-4]+'_bic.tif', mode='CMYK')
            self.save_img(ms_image.cpu().data, name[0][0:-4]+'_gt.tif', mode='CMYK')
            self.save_img(prediction.cpu().data, name[0][0:-4]+'.tif', mode='CMYK')
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))
        # print(name)
        # ref_results,no_ref_results = cal_performance(os.path.join(self.cfg['test']['save_dir'], self.cfg['test']['type']+ "_" + str(self.cfg['stage']) + 'stage', str(self.now_time)))
        # if float(self.best_ref_results['deep   '][0]) < float(ref_results['deep   '][0]):
        #     self.best_ref_results = ref_results
        #     self.best_no_ref_results = no_ref_results
            
        #     super(Solver, self).save_checkpoint()
        #     self.ckp['net'] = self.model.state_dict()
        #     self.ckp['optimizer'] = self.optimizer.state_dict()
        #     if not os.path.exists(self.cfg['checkpoint'] + '/' + str(self.log_name)):
        #         os.mkdir(self.cfg['checkpoint'] + '/' + str(self.log_name))
        #     torch.save(self.ckp, os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'Best_best.pth'))
    
    def save_img(self, img, img_name, mode):
        save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
        # save img
        save_dir = os.path.join(self.cfg['test']['save_dir'], self.cfg['test']['type']+ "_" + str(self.cfg['stage']) + 'stage', str(self.now_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_fn = save_dir +'/'+ img_name
        save_img = np.uint8(save_img*255).astype('uint8')
        save_img = Image.fromarray(save_img, mode)
        save_img.save(save_fn)

    def check_gpu(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            cudnn.benchmark = True
              
            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >=0:
                    self.gpu_ids.append(gid)

            torch.cuda.set_device(self.gpu_ids[0]) 
            self.loss = self.loss.cuda(self.gpu_ids[0])
            #self.vggloss = self.vggloss.cuda(self.gpu_ids[0])
            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids) 

    def check_pretrained(self):
        checkpoint = os.path.join(self.cfg['pretrain']['pre_folder'], self.cfg['pretrain']['pre_sr'])
        if os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['net'])
            self.epoch = torch.load(checkpoint, map_location=lambda storage, loc: storage)['epoch']
            if self.epoch > self.nEpochs:
                raise Exception("Pretrain epoch must less than the max epoch!")
        else:
            raise Exception("Pretrain path error!")

    def save_checkpoint(self):
        super(Solver, self).save_checkpoint()
        self.ckp['net'] = self.model.state_dict()
        self.ckp['optimizer'] = self.optimizer.state_dict()
        if not os.path.exists(self.cfg['checkpoint'] + '/' + str(self.log_name)):
            os.mkdir(self.cfg['checkpoint'] + '/' + str(self.log_name))
        torch.save(self.ckp, os.path.join(self.log_dir+self.log_name, 'weights_latest_x.pth'))

        if self.cfg['save_best']:
            if self.records['SSIM'] != [] and self.records['SSIM'][-1] == np.array(self.records['SSIM']).max():
                shutil.copy(os.path.join(self.log_dir+self.log_name, 'weights_latest_x.pth'),
                            os.path.join(self.log_dir+self.log_name, 'weights_best_x.pth'))

    def run(self):
        self.check_gpu()
        if self.cfg['pretrain']['pretrained']:
            self.check_pretrained()
        try:
            while self.epoch <= self.nEpochs:
                self.train()
                self.eval()

                self.save_checkpoint()
                print('best_ref_results', self.best_ref_results['deep   '])

                self.epoch += 1
        except KeyboardInterrupt:
            self.save_checkpoint()
        save_config(self.log_dir+self.log_name, 'Training done.')

    def _add_image(self, name, image, epoch):
        if image.size(0) == 4:
            image = torch.permute(image, (1, 2, 0))
            aux = torch.ones(image.size(0), image.size(1), 3)
            aux[:, :, 0] = image[:, :, 2]
            aux[:, :, 1] = image[:, :, 1]  # OKEY
            aux[:, :, 2] = image[:, :, 0]
            image = aux
            image = torch.permute(image, (2, 0, 1))
        elif image.size(0) > 4:
            image = image.cpu()
            image = hiperspectral_to_rgb(image)
            image = torch.permute(image, (2, 0, 1))
        image = self._scaleminmax(image)
        self.writer.add_image(name, image, global_step=epoch)

    @staticmethod
    def _scaleminmax(v, new_min=0, new_max=1):
        v_min, v_max = v.min(), v.max()
        return (v - v_min) / (v_max - v_min) * (new_max - new_min) + new_min
