# -*- coding: utf-8 -*-
"""
@author: losstie
"""
import torch
import numpy as np
from experiment.base_experiment import BaseExperimentMaker


class GANExperimentMaker(BaseExperimentMaker):
    def __init__(self, Gan, opt, recorder=None):
        super(GANExperimentMaker, self).__init__(opt, recorder)
        self.gan = Gan
        import os
        print(self.device, os.environ['CUDA_VISIBLE_DEVICES'])
        self.G = self.gan.G.to(self.device)
        self.D = self.gan.D.to(self.device)

        print('Parameter nums of G:{}'.format(sum(param.numel() for param in self.G.parameters())))
        print('Parameter nums of D:{}'.format(sum(param.numel() for param in self.D.parameters())))

        # optimizer & criterion
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=opt.lr)
        self.D_optimizer = torch.optim.Adam(self.G.parameters(), lr=opt.lr)

    # def a_step(self, input, label, epoch, fortrain=True, lambda_ = 1):
    #     _input = [item.to(self.device) for item in input]
    #     _label = label.to(self.device)

    #     # training Generator
    #     fake_img = self.G(*_input)
    #     pred_fake = self.D(torch.cat((_input[0], fake_img), dim=1))
    #     g_loss = self.gan.G_criterion(fake_img, _label, pred_fake)

    #     if fortrain:
    #         self.G_optimizer.zero_grad()
    #         g_loss.backward()
    #         # nn.utils.clip_grad_norm(self.G.parameters(), 0.4) #姊害瑁佸壀
    #         self.G_optimizer.step()

    #         # training Discriminator
    #         self.D_optimizer.zero_grad()
    #         pred_fake = self.D(torch.cat((_input[0], fake_img.detach()), dim=1))
    #         pred_real = self.D(torch.cat((_input[0], _label), dim=1))
    #         d_loss = self.gan.D_criterion(pred_fake, pred_real)

    #         d_loss.backward()
    #         # nn.utils.clip_grad_norm(self.D.parameters(), 0.4) #姊害瑁佸壀
    #         self.D_optimizer.step()

    #     return g_loss, fake_img
    def a_step(self, input, label, epoch, fortrain=True, lambda_=1):
        _input = [item.to(self.device) for item in input]
        _label = label.to(self.device)

        fake_img = self.G(*_input)
        fake_diff1, fake_diff2 = self.getMS_diff(fake_img)
        pred_fake = self.D(torch.cat((_input[0],_input[1],_input[2], fake_img.to(self.device), fake_diff1.to(self.device), fake_diff2.to(self.device)), dim=1))
        g_loss = self.gan.G_criterion(fake_img, _label, pred_fake,self.getMS_diff)

        if fortrain:
            self.G_optimizer.zero_grad()
            g_loss.backward()
            # nn.utils.clip_grad_norm(self.G.parameters(), 0.4) #姊害瑁佸壀
            self.G_optimizer.step()

            # training Discriminator
            self.D_optimizer.zero_grad()
            pred_fake = self.D(torch.cat((_input[0],_input[1],_input[2], fake_img.detach().to(self.device), fake_diff1.to(self.device), fake_diff2.to(self.device)), dim=1))
            real_diff1, real_diff2 = self.getMS_diff(_label)
            pred_real = self.D(torch.cat((_input[0],_input[1],_input[2], _label.to(self.device), real_diff1.to(self.device), real_diff2.to(self.device)), dim=1))
            d_loss = self.gan.D_criterion(pred_fake, pred_real)

            d_loss.backward()
            # nn.utils.clip_grad_norm(self.D.parameters(), 0.4) #姊害瑁佸壀
            self.D_optimizer.step()
        return g_loss, fake_img

    def getMS_diff(self,ms):
        A1 = -(np.eye(128, k=-127) + np.eye(128, k=1)) + np.eye(128)
        A2 = -(np.eye(128, k=-126) + np.eye(128, k=2)) + np.eye(128)
        A1 = A1.astype(np.float32)[np.newaxis,:,:]
        A2 = A2.astype(np.float32)[np.newaxis,:,:]
        A_1 = A1
        A_2 = A2

        for i in range(ms.shape[1]-1):
            A_1 = np.r_[A_1,A1]
            A_2 = np.r_[A_2,A2]
        ms = ms.data.cpu().numpy()
        ms_diff1_h = np.matmul(ms[0,:,:,:].reshape(ms.shape[1],ms.shape[2],ms.shape[3]),A_1)[np.newaxis,:,:,:]

        for i in range(1, ms.shape[0]):
            ms_diff1_h = np.r_[ms_diff1_h,np.matmul(ms[i,:,:,:].reshape(ms.shape[1],ms.shape[2],ms.shape[3]),A_1)[np.newaxis,:,:,:]]

        ms_diff1_v = np.matmul(A_1, ms[0,:,:,:].reshape(ms.shape[1], ms.shape[2],ms.shape[3]))[np.newaxis,:, :,:]
        for i in range(1, ms.shape[0]):
            ms_diff1_v = np.r_[ms_diff1_v, np.matmul(A_1, ms[i,:, :, :].reshape(ms.shape[1], ms.shape[2],ms.shape[3]))[np.newaxis,:, :,:]]

        ms_diff2_h = np.matmul(ms[0,:, :, :].reshape(ms.shape[1], ms.shape[2],ms.shape[3]), A_2)[np.newaxis,:, :,:]
        for i in range(1, ms.shape[0]):
            ms_diff2_h = np.r_[ms_diff2_h, np.matmul(ms[i,:, :, :].reshape(ms.shape[1], ms.shape[2],ms.shape[3]), A_2)[np.newaxis,:, :, :]]

        ms_diff2_v = np.matmul(A_2, ms[0,:, :,:].reshape(ms.shape[1], ms.shape[2],ms.shape[3]))[np.newaxis,:, :, :]
        for i in range(1, ms.shape[0]):
            ms_diff2_v = np.r_[ms_diff2_v, np.matmul(A_2, ms[i,:, :, :].reshape(ms.shape[1], ms.shape[2],ms.shape[3]))[np.newaxis,:, :, :]]

        return torch.from_numpy(ms_diff1_h + ms_diff1_v), torch.from_numpy(ms_diff2_h + ms_diff2_v)
