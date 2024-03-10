#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:07:03
LastEditTime: 2020-11-25 19:24:54
@Description: file content
'''
import os, torch, time
from utils.utils import save_config
from data.dataset import data
from data.data import get_data,get_test_data
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class BaseSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nEpochs = cfg['nEpochs']
        self.checkpoint_dir = cfg['checkpoint']
        self.epoch = 1

        self.timestamp = int(time.time())

        if cfg['gpu_mode']:
            self.num_workers = cfg['threads']
        else:
            self.num_workers = 0

        self.train_dataset = get_data(cfg, cfg['data_dir_train'])
        self.train_loader = DataLoader(self.train_dataset, cfg['data']['batch_size'], shuffle=False,
            num_workers=self.num_workers)
        self.val_dataset = get_data(cfg, cfg['data_dir_eval'])
        self.val_loader = DataLoader(self.val_dataset, cfg['data']['batch_size'], shuffle=False,
            num_workers=self.num_workers)
        self.test_dataset = get_test_data(cfg, cfg['data_dir_eval'])
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=4,
                                      num_workers=self.num_workers)
        self.records = {'Epoch': [], 'PSNR': [], 'SSIM': [], 'Loss': [],'QNR':[],'D_lamda':[],'D_s':[]}

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
