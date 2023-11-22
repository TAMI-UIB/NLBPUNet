
import torch
from torch import nn
import torch.nn.functional as F
import math
import cv2 as cv
import numpy as np
from functools import reduce
# import random
from torch.nn.functional import cosine_similarity
from experiment.base_experiment import BaseExperimentMaker
eps = 1e-8


class CNNExperimentMaker(BaseExperimentMaker):
    def __init__(self, netWork, opt, recorder=None):
        super(CNNExperimentMaker, self).__init__(opt, recorder)
        # assign model, and put it to device
        self.model = netWork.to(self.device)
        print('Parameter nums of model:{}'.format(sum(param.numel() for param in self.model.parameters())))

        # optimizer & criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=opt.lr_descentfactor)
        # self.ldf = opt.lambda_descentfactor
        # self.lambda_ = opt.lambda_weight
        self.block_size = opt.block_size
        self.channels = 4 if opt.dataset=='gf2' else 8
        self.net_name = opt.net
        self.l1 = nn.L1Loss(reduction='mean')
        self.l2 =  nn.MSELoss(reduction='mean')
        self.cos = nn.CosineSimilarity()
        self.mseLoss = nn.MSELoss()

    def a_step(self, input, label, epoch, fortrain=True, lambda_=1):
        _input = [item.to(self.device) for item in input]
        _label = label.to(self.device)

        output = self.model(*_input)

        if self.net_name == 'bdpn':
            loss = self.bdpnLoss(output, _label, epoch,lambda_)
            output = output if not isinstance(output, list) else output[-1]
        elif self.net_name == 'mhfnet':
            loss = self.mhfLoss(output, _label)
        elif self.net_name == 'pmfnet':
            loss = self.pmfLoss(output, _label)            
        elif self.net_name == 'vpnet':
            loss = self.vpLoss(output, _label)
            output = [output[0][-1]]
        elif self.net_name in ['mddl','gfrnet', 'qrnetb', 'pcnn', 'fusionnet', 'nlunet']:
            loss = self.l1(output, _label)
        elif self.net_name == 'nlrnet':
            loss = self.criterion(output, _label)
        elif self.net_name in ['dgfcnn','pmsrn','dircnn']:
            loss = self.dgfcnnLoss(output, _label)
        else:
            loss = self.l2(output, _label)

        if fortrain:
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(self.model.parameters(), 0.4)
            self.optimizer.step()
            if self.net_name == 'bdpn':
                self.scheduler.step()

        return loss, output

    def mhfLoss(self, output, _label):
        out, out_ls, E = output
        #out, out_ls, E, ms, S = output
        l1 = self.l1(out, _label)
        # l2 = 0.5*self.l2(YA, _label)
        #l2 = 1*self.l2((ms+S), _label)
        l3 = self.l1(E, torch.zeros_like(E).to('cuda'))
        #loss = l1 + l2 + l3
        loss = l1 + l3
        #loss = l1
        for item in out_ls:
            loss = loss + 0.1*self.l1(item, _label)
        return loss

    #def pmfLoss(self, output, _label):
        #out, E1, E2, stage_ls = output
        #out, out_ls, E, ms, S = output
        #l0 = 7*self.l2(out, _label)
        #l0 += sum(map(lambda x: self.l2(x, _label), stage_ls))
        # l2 = 0.5*self.l2(YA, _label)
        #l2 = 1*self.l2((ms+S), _label)
        #l1 = 0.1*self.l2(E1, torch.zeros_like(E1).to('cuda'))
        #l2 = 0.1*self.l2(E2, torch.zeros_like(E2).to('cuda'))
        #loss = l1 + l2 + l3
        #loss = l0 + l1+ l2
        #loss = l1

        # return loss   
        
    def pmfLoss(self, output, _label):
        out, E1, E2 = output
        #out, out_ls, E, ms, S = output
        l0 = self.l1(out, _label)
        #l0 += sum(map(lambda x: self.l2(x, _label), stage_ls))
        # l2 = 0.5*self.l2(YA, _label)
        #l2 = 1*self.l2((ms+S), _label)
        l1 = 0.1*self.l1(E1, torch.zeros_like(E1).to('cuda'))
        l2 = 0.1*self.l1(E2, torch.zeros_like(E2).to('cuda'))
        #loss = l1 + l2 + l3
        loss = l0 + l1+ l2
        #loss = l1

        return loss 

    def vpLoss(self, output, _label):
        out_ls, sym_ls, q = output
        #out, out_ls, E, ms, S = output
        #l1 = 10*self.l2(out, _label)
        # l2 = 0.5*self.l2(YA, _label)
        #l2 = 1*self.l2((ms+S), _label)
        #l3 = 0.1*self.l2(sym, torch.zeros_like(sym).to('cuda'))
        #loss = l1 + l2 + l3
        loss = 0
        #loss = l1
        for item in out_ls:
            loss = loss+10*self.l2(item, _label)
        #for sym in sym_ls:
          #loss = loss+0.1*torch.mean(torch.square(sym))
        return loss

    def bdpnLoss(self, output, _label, epoch, lambda_):
        output_1, output_2 = output
        dsize = int(self.block_size*0.5)
        _label_1 = F.interpolate(_label, size=[self.block_size//2, self.block_size//2], mode='bilinear')

        loss1 = self.bdpnHelpLoss(output_1, _label_1)
        loss2 = self.bdpnHelpLoss(output_2, _label)

        # if self.lambda_ >= 0.01:
        #     self.lambda_ = self.lambda_ - *self.ldf

        return lambda_*loss1 + (1-lambda_)*loss2

    def bdpnHelpLoss(self, pre, ref):
        B = self.channels
        a = 0
        for b in range(B):
            rmse_b = torch.sqrt(self.mseLoss(pre[:,b,:,:], ref[:,b,:,:]))
            e_ui = math.exp(-torch.mean(ref[:,b,:,:]))
            tmp = (rmse_b * e_ui)**2
            a += tmp
        return torch.sqrt(a/B)

    def _downsample(self, img, dsize, ksize=(7, 7), interpolation=cv.INTER_AREA):
        img = img.numpy()
        blur = cv.GaussianBlur(img, ksize, 0)
        downsampled = cv.resize(blur, dsize, interpolation=interpolation)
        return torch.from_numpy(downsampled)

    def criterion(self, output, _label):
        spatital_loss = self.l1(output, _label) * 85
        spectral_loss = torch.mean(1 - cosine_similarity(output, _label, dim=1)) * 15

        # band shuffle
        sq = torch.Tensor([1, 2, 3, 0]).type(torch.LongTensor)
        # shuffle real_img
        base = _label[:, sq, :, :]
        new_label = _label - base
        # shuffle fake_img
        base = output[:, sq, :, :]
        new_fake = output - base
        spectral_loss2 = self.l1(new_label, new_fake) * 15

        return spatital_loss + spectral_loss + spectral_loss2

    def FrobeniusLoss(self, output, _label):
        loss_front = nn.MSELoss(reduction='sum')
        return torch.sqrt(loss_front(output, _label)+eps)
    def dgfcnnLoss(self, output, _label):
        spectral_loss = self.l1(output, _label)* 90
        real_diff1 = self.getMS_diff(_label)
        pred_diff1 = self.getMS_diff(output)
        spatial_loss = torch.mean(torch.abs(real_diff1 - pred_diff1)) * 80
        return spectral_loss + spatial_loss
    def getMS_diff(self,ms):
        A1 = -(np.eye(128, k=-127) + np.eye(128, k=1)) + np.eye(128)
        A1 = A1.astype(np.float32)[np.newaxis,:,:]
        A_1 = A1
        for i in range(ms.shape[1]-1):
            A_1 = np.r_[A_1,A1]
        ms = ms.data.cpu().numpy()
        ms_diff1_h = np.matmul(ms[0,:,:,:].reshape(ms.shape[1],ms.shape[2],ms.shape[3]),A_1)[np.newaxis,:,:,:]
        for i in range(1, ms.shape[0]):
            ms_diff1_h = np.r_[ms_diff1_h,np.matmul(ms[i,:,:,:].reshape(ms.shape[1],ms.shape[2],ms.shape[3]),A_1)[np.newaxis,:,:,:]]
        ms_diff1_v = np.matmul(A_1, ms[0,:,:,:].reshape(ms.shape[1], ms.shape[2],ms.shape[3]))[np.newaxis,:, :,:]
        for i in range(1, ms.shape[0]):
            ms_diff1_v = np.r_[ms_diff1_v, np.matmul(A_1, ms[i,:, :, :].reshape(ms.shape[1], ms.shape[2],ms.shape[3]))[np.newaxis,:, :,:]]

        return torch.from_numpy(ms_diff1_h + ms_diff1_v)
