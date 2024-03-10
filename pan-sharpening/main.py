#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-12 23:50:07
@LastEditTime: 2020-06-23 17:46:46
@Description: main.py
'''

from utils.config import get_config
from solver.unisolver import Solver
# from solver.gf_solver import Solver
# from solver.midnsolver import Solver
# from solver.innformersolver import Solver
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N_SR')
    parser.add_argument('--option_path', type=str, default='/home/hexuanhua/src/sr/pan-sharpening/option.yml')
    opt = parser.parse_args()
    cfg = get_config(opt.option_path)
    solver = Solver(cfg)
    solver.run()
    