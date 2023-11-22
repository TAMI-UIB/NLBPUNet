#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-12 23:50:07
@LastEditTime: 2020-06-23 17:46:46
@Description: main.py
'''

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()
sys.path.extend([os.environ['PROJECT_PATH']])
from sota.MDCUN.solver.solver import Solver
from sota.MDCUN.utils.config import get_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N_SR')
    parser.add_argument('--option_path', type=str, default=os.environ['PROJECT_PATH']+'/sota/MDCUN/option_wv2.yml')
    opt = parser.parse_args()
    cfg = get_config(opt.option_path)
    solver = Solver(cfg)
    solver.run()
