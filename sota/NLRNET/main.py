import argparse
import pickle
import time
import os
from torch import save, load
from os.path import join
from utils.recorder import Recorder

# CNN
from net import NLRNet
from experiment import TestParameter, environment_init
from experiment import CNNExperimentMaker, GANExperimentMaker
import os
# Training Setting
parser = argparse.ArgumentParser(description="Pansharpening experiment")
parser.add_argument("--mode", type=str, default='test',choices=['train','test','local'], help='enter the  mode')
parser.add_argument("--dataset", type=str, default='wv2', choices=['gf2', 'wv2', 'wv'], help='the dataset for experiment')
parser.add_argument("--fr", type=bool, default=False, help="whether is full resolution experiment")
parser.add_argument("--resource",type=str, default=r'resource/{}', help=" the resource path")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
parser.add_argument("--epoch_nums", type=int, default=40, help="the max epochs for train")
parser.add_argument("--cuda", type=str, default='0', help="Use cuda? & device number, -1 note reject")
parser.add_argument("--block_size", type=int, default=128, help="Size of cuted blocks")
parser.add_argument("--seed", type=int, default=1024, help="Random seed , useful on guaranting same result")
parser.add_argument("--net", type=str, default='nlrnet', help='Training Networks')
parser.add_argument("--step_print", type=int, default=50, help='Print training loss per step')
parser.add_argument("--persist", type=str, default='1', help='Cache intermediate results into file')
parser.add_argument("--test_img", type=str, default='6', help='test data number (1-9)')
parser.add_argument("--warm_start", action='store_true', default=False, help='start without preprocessing data')
parser.add_argument("--pretrained", type=bool, default=False, help='load the pretrained model')
parser.add_argument("--noref", action='store_true', default=False, help='assement with no reference')
parser.add_argument("--eval_step", type=int, default=1, help="call the evaluation function per eval_step epoch")
parser.add_argument("--lr", type=float, default=0.0005, help="Learning Rate, Default=0.1")
parser.add_argument("--lr_descentfactor", default=0.8, help="the descent factor was set 0.8 every 100 epochs")
parser.add_argument("--lambda_weight", type=int, default=1, help="weight of l1 loss of the generator loss")
parser.add_argument("--lambda_descentfactor", type=int, default=0.0005, help="the coefficient lambda was decreased by 0.01 every epochs")
parser.add_argument("--gan_weight", type=int, default=1, help="weight of adversal loss of the generator loss")
# parser.add_argument("--spectral_weight", type=int, default=90, help="weight of adversal loss of the generator loss")
# parser.add_argument("--spatial1_weight", type=int, default=80, help="weight of adversal loss of the generator loss")
# parser.add_argument("--spatial2_weight", type=int, default=40, help="weight of adversal loss of the generator loss")#40
parser.add_argument("--l1_weight", type=int, default=85, help="weight of l1 loss of the generator loss")
parser.add_argument("--model", type=str, default=r'model/nlunet_wv2/')

start = time.time()
opt = parser.parse_args()

# init the program's environment
environment_init(opt.seed, opt.cuda)

# Networks & Preprocessing Method
nets = {'nlrnet':NLRNet}

preprocessing = {'nlrnet':'nlrnet'}

assert opt.net in nets.keys(), '--net , spelling mistakes !'

if 'gan' not in opt.net:
    NetWork = nets[opt.net](opt.dataset)
else:
    # TODO the program related gan maybe exist problem
    NetWork = nets[opt.net](opt)
recorder = Recorder(output_dir=opt.model)

if opt.pretrained:
    assert os.path.exists(join(opt.model,'net.model')), '--model, not exist!'
    NetWork.load_state_dict(load(join(opt.model, 'net.model')))
    with open(join(opt.model, 'recorder.model'), 'rb') as fb:
        recorder = pickle.load(fb)

# if the opt not in preprocssing dict, return mddl. In a word, default pannet
opt.prep_method = preprocessing.get(opt.net, 'mddl')

# Local test setting
test_opt = {'prep_method': opt.prep_method, 'net': opt.net, 'warm_start': False, 'epoch_nums': 1, 'batch_size': opt.batch_size,
            'lr': 0.0001, 'lr_descentfactor':0.8,'block_size': 64, 'step_print': 5, 'seed': 1024, 'persist': 0,
            'noref': True,'model': opt.model, 'cuda': '-1', 'eval_step': 1, 'test_img': '19', 'lambda_weight': 1,
            'lambda_descentfactor':0.001, 'dataset': opt.dataset, 'resource':opt.resource, 'mode':opt.mode, 'fr':opt.fr}

if opt.mode =='local':
    print('Enter local mode:')
    opt = TestParameter(test_opt)

print('Experiment setting:\n', opt)

# Environment Init
# setting random seed1
# guaranteeing same result between different experiments with same option, at least on same device.
# decided the platform the code would run, gup or cpu, and cuda number if gpu available.
# environment_init(opt.seed, opt.cuda)

# Make Experiment With CNN or Gan
if 'gan' not in opt.net:
    exper = CNNExperimentMaker(NetWork, opt, recorder)
else:
    exper = GANExperimentMaker(NetWork, opt, recorder)

if opt.mode != 'test':
    # train
    exper.start()
    # save the model
    save(NetWork.state_dict(), join(opt.model, 'net.model'))
else:
    # generate img
    # if pretrained is not true, load the model
    if not opt.pretrained:
        assert os.path.exists(join(opt.model,'net.model')), '--model, not exist!'
        NetWork.load_state_dict(load(join(opt.model, 'net.model')))
        with open(join(opt.model, 'recorder.model'), 'rb') as fb:
            recorder = pickle.load(fb)

    img = exper.rm.test_img_restore(exper.evaluate_network(), fr=opt.fr)
    exper.cache_img(img)
end = (time.time() - start) / 60
print("total %d h %d m" % (end // 60, end % 60))
