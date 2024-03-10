#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2021-03-20 14:44:14
LastEditTime: 2021-03-22 15:25:02
Description: file content
'''
import  sys
sys.path.insert(1, "/ghome/fuxy/DPFN-master/thop")
sys.path.insert(1, "/ghome/fuxy/DPFN-master/ptflops")
sys.path.insert(1, "/ghome/fuxy/DPFN-master/torchsummaryX")

from thop import profile
import importlib, torch
from utils.config import get_config
import math
from  ptflops import get_model_complexity_info
import time

if __name__ == "__main__":
    model_name = 'panformer'
    net_name = model_name.lower()
    lib  = importlib.import_module('model.' + net_name)
    net = lib.Net
    cfg = get_config('option5.yml')
    model = net(
            num_channels=5,
            base_filter=64,
            args=cfg
    )
    input = torch.randn(1, 4, 32, 32)
    input1 = torch.randn(1, 1, 128, 128)
    input2 = torch.randn(1, 4, 128, 128)


    # macs, params = get_model_complexity_info(model, ((4, 32, 32), (), (1, 128, 128)),
    #                                          as_strings=True,print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # import torchsummaryX
    # torchsummaryX.summary(model, [input.cpu(), None, input1.cpu()])

    # print("The torchsummary result")
    # from torchsummary import summary
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # summary(model.cuda(), [(4, 32, 32), (), (1, 128, 128)])
    #
    print("The thop result")
    flops, params = profile(model, inputs=(input, input2, input1))
    print('flops:{:.6f}, params:{:.6f}'.format(flops/(1e9), params/(1e6)))