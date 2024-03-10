#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2021-03-20 14:44:14
LastEditTime: 2021-03-22 15:25:02
Description: file content
'''
import  sys
# sys.path.insert(1, "/ghome/fuxy/DPFN-master/thop")
# sys.path.insert(1, "/ghome/fuxy/DPFN-master/ptflops")
# sys.path.insert(1, "/ghome/fuxy/DPFN-master/torchsummaryX")

from thop import profile
import importlib, torch
from utils.config import get_config
import math
#from  ptflops import get_model_complexity_info
import time

if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'pan_inn9' #1.414   0.086    #hmb 57.515606   2.155652
    net_name = model_name.lower()
    lib  = importlib.import_module('model.' + net_name)
    net = lib.Net
    cfg = get_config('option.yml')
    # model = net(
    #     ms_channels=4,
    #     pan_channels=1,
    #     n_feat=16,
    # ).cuda(0)
    # model = net(
    #     ms_channels=4,
    #     pan_channels=1,
    #     n_feat=8
    # ).cuda(0)
    model = net(
            num_channels=4,
            base_filter=32,
            args=cfg
    ).to("cuda")
    device="cuda"
    # model = net(
    scale=1
#     input = torch.randn(1, 4, 32, 32).cuda()
#     input1 = torch.randn(1, 1, 128, 128).cuda()
#     input2 = torch.randn(1, 4, 128, 128).cuda()
    input = torch.randn(1, 4, 32*scale, 32*scale).to(device)
    input1 = torch.randn(1, 1, 128*scale, 128*scale).to(device)
    input2 = torch.randn(1, 4, 128*scale, 128*scale).to(device)
    model.eval()
    torch.cuda.reset_max_memory_allocated(device)

#     mem_before = torch.cuda.memory_allocated(device)
    with torch.no_grad():
        model(input, input2, input1)
        mem_after = torch.cuda.memory_allocated(device)
        max_mem_used_during_forward_pass = torch.cuda.max_memory_allocated(device)

    print(f"Memory used by the model: {max_mem_used_during_forward_pass/1024 ** 3} G")

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
    # flops, params = profile(model, inputs=(input, input2, input1))
    flops, params = profile(model, inputs=(input, input2,input1))

    print('flops:{:.6f}, params:{:.6f}'.format(flops/(1e9), params/(1e6)))


    import time
    with torch.no_grad():
        model(input, input2, input1)
        start = time.time()
        for i in range(100):
            model(input, input2, input1)
        end = time.time()
    elapsed_time_in_seconds = end - start
    elapsed_time_in_milliseconds = elapsed_time_in_seconds * 1000/100

    print("time: ", elapsed_time_in_milliseconds)
