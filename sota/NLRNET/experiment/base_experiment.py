# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from processing import ResourceManager, GenerDataSet
from utils.tools import sam, ergas, scc, D_lambda, D_s, qindex
from abc import abstractmethod


class TestParameter(object):
    def __init__(self, parameters):
        names = self.__dict__
        for k, v in parameters.items():
            names[k] = v

    def __str__(self):
        head = 'NameSpace('
        for k, v in self.__dict__.items():
            head += '{}={}, '.format(k, v)
        return head[:-2] + ')'


def environment_init(random_seed, cuda):
    np.random.seed(random_seed)
    if cuda == -1:
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BaseExperimentMaker(object):
    def __init__(self, opt, recorder=None):
        self.model = None
        self.G = None
        self.D = None

        self.eval_step = opt.eval_step
        self.epoch_nums = opt.epoch_nums
        self.step_print = opt.step_print
        self.persist_to = opt.persist
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.noref = opt.noref
        self.ldf = opt.lambda_descentfactor
        self.lambda_ = opt.lambda_weight
        self.fr = opt.fr

        # load
        self.load_data(opt.block_size, opt.prep_method, opt.test_img, opt.warm_start, opt.seed, opt.batch_size, opt.dataset, opt.resource)

        # record
        self.recorder = recorder

    def load_data(self, block_size, prep_method, test_img, warm_start, random_seed, batch_size, dataset, resource):
        # load data
        self.rm = ResourceManager(block_size, network=prep_method,
                                  test_img='record_{}.mat'.format(test_img),
                                  warm_start=warm_start, seed=random_seed, dataset=dataset, resource=resource)
        train_dataset = GenerDataSet(self.rm, dtype='train')
        self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        # add something to net
        self.volunteers_nums = batch_size - len(self.rm.test_blocks) % batch_size
        if self.volunteers_nums != batch_size:
            self.rm.test_blocks += self.rm.test_blocks[:self.volunteers_nums]

        # TODO merge the fr and res experiment test loader
        test_dataset = GenerDataSet(self.rm, dtype='test')
        self.test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        # add something to net
        self.fr_volunteers_nums = batch_size - len(self.rm.fr_test_blocks) % batch_size
        if self.fr_volunteers_nums != batch_size:
            self.rm.fr_test_blocks += self.rm.fr_test_blocks[:self.fr_volunteers_nums]

        fr_test_dataset = GenerDataSet(self.rm, dtype='fr_test')
        self.fr_test_data_loader = DataLoader(dataset=fr_test_dataset, batch_size=batch_size, shuffle=False)

    def process_output(self, collect_output, collect_label, vol_nums):
        # remove rest output
        if vol_nums != 0:

            collect_output[-1] = collect_output[-1][-vol_nums:, :, :, :]
            collect_label[-1] = collect_label[-1][-vol_nums:, :, :, :]

        # condat all batchsize
        collect_output = np.row_stack(np.array(collect_output))
        collect_label = np.row_stack(np.array(collect_label))

        # slice net output
        collect_output[collect_output < 0] = 0
        collect_output[collect_output > 1] = 1

        return collect_output, collect_label

    def ref_assement(self, collect_output, collect_label):
        """down_resolution index test"""
        h, w, c = collect_output[0].shape
        element_nums = h * w * c
        _sam, _erags, _scc, _qn = [], [], [], []
        for item in range(collect_output.shape[0]):
            label_block = collect_label[item, :, :, :]
            output_block = collect_output[item, :, :, :]

            if len(label_block[label_block == 0]) / element_nums > 0.5:
                continue

            _sam.append(sam(label_block, output_block))
            _erags.append(ergas(output_block, label_block))
            _scc.append(scc(output_block, label_block))
            _qn.append(qindex(output_block, label_block, block_size=128))
        self.recorder.write_assment_ref(_sam, _erags, _scc, _qn)

    def noref_assement(self, output, alpha=1, beta=1):
        """full_resolution index test"""
        fr_fake = self.rm.test_img_restore(output, fr=True)
        D_lambda_idx = D_lambda(self.rm.fr_test_lrms, fr_fake)
        D_s_idx = D_s(self.rm.fr_pan[:, :, 0], self.rm.fr_test_lrms, fr_fake)
        QNR_idx = (1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta
        self.recorder.write_assment_noref(QNR_idx, D_lambda_idx, D_s_idx)

    def cache_img(self, test_img):
        print("start cache img")
        with open('cache/img.pickle', 'wb') as fb:
            pickle.dump({'img': test_img}, fb)
        print('end')

    def iteration(self, epoch, total_step, fortrain=True):
        step_loss = []
        for step, (input, label) in enumerate(self.train_data_loader):
            loss, _ = self.a_step(input, label, epoch, fortrain, self.lambda_)
            step_loss.append(loss.cpu().item())

            if fortrain and (step % self.step_print == 0):
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, self.epoch_nums, step + 1, total_step, torch.sum(loss).item()))
        return np.mean(step_loss)

    def start(self):
        self.update_model_statu('train')

        total_step = len(self.train_data_loader)
        for epoch in range(self.epoch_nums):
            # self.adjust_learning_rate(self.optimizer, epoch) #adjust learning_rate
            self.lambda_ = self.lambda_ if epoch==0 else (self.lambda_ - self.ldf)
            epoch_loss = self.iteration(epoch, total_step, fortrain=True)
            self.recorder.write_loss(train_loss=epoch_loss, _print=False)

            if (epoch + 1) % self.eval_step == 0:
                output = self.evaluate_network()
                # if self.save_img:  # should saving best model !!!!!
                #     test_image = self.rm.test_img_restore(output, fr=True)
                #     self.cache_img(test_image)
            self.recorder.update_epoch()

    def evaluate_network(self):
        if not self.fr:
            collect_output, collect_label = [], []
            print("resdiual")
            self.update_model_statu('eval')
            with torch.no_grad():
                step_loss_ls = []
                for step, (input, label) in enumerate(self.test_data_loader):
                    step_loss, output = self.a_step(input, label, self.epoch_nums, fortrain=False, lambda_=self.lambda_)
                    step_loss_ls.append(step_loss.cpu().item())
                    
                    # collect_output.append(torch.Tensor.permute(output[-1][0], [0, 2, 3, 1]).cpu().numpy())
                    # collect_output.append(torch.Tensor.permute(output[0], [0, 2, 3, 1]).cpu().numpy())
                    collect_output.append(torch.Tensor.permute(output, [0, 2, 3, 1]).cpu().numpy())
                    collect_label.append(torch.Tensor.permute(label, [0, 2, 3, 1]).numpy())
                    #collect_output = collect_label
            self.update_model_statu('train')
            output, label = self.process_output(collect_output, collect_label, self.volunteers_nums)
            self.ref_assement(output, label)
        else:
            print("full")
            collect_output = []
            self.update_model_statu('eval')
            with torch.no_grad():
                for step, (input, label) in enumerate(self.fr_test_data_loader):
                    _, output = self.a_step(input, label, self.epoch_nums, fortrain=False, lambda_=self.lambda_)
                    # collect_output.append(torch.Tensor.permute(output[0], [0, 2, 3, 1]).cpu().numpy())
                    collect_output.append(torch.Tensor.permute(output, [0, 2, 3, 1]).cpu().numpy())
            self.update_model_statu('train')
            output, _ = self.process_output(collect_output, collect_output, self.fr_volunteers_nums)

            self.noref_assement(output)

        # if self.noref:
        #     collect_output = []
        #     self.update_model_statu('eval')
        #     with torch.no_grad():
        #         for step, (input, label) in enumerate(self.fr_test_data_loader):
        #             _, output = self.a_step(input, label, self.epoch_nums, fortrain=False, lambda_=self.lambda_)
        #             collect_output.append(torch.Tensor.permute(output, [0, 2, 3, 1]).cpu().numpy())
        #     self.update_model_statu('train')
        #     fr_output, _ = self.process_output(collect_output, collect_output, self.fr_volunteers_nums)

            #self.noref_assement(fr_output)

        #self.recorder.write_loss(test_loss=np.mean(step_loss_ls))
        # if self.noref:
        #     return fr_output
        # else:
        #     return output
        return output


    def update_model_statu(self, status='train'):
        for item in [self.model, self.G, self.D]:
            if item is not None:
                if status == 'train':
                    item.train()
                else:
                    item.eval()

    @abstractmethod
    def a_step(self, *args, **kwargs):
        pass
