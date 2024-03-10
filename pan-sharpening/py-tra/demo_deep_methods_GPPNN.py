# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
"""

import numpy as np
import cv2
import os
from scipy import signal
from metricsutil import get_metrics_reduced
#torch.from_numpy
#torch.float()

from methods.Bicubic import Bicubic
from methods.Brovey import Brovey
from methods.PCA import PCA
from methods.IHS import IHS
from methods.SFIM import SFIM
from methods.GS import GS
from methods.Wavelet import Wavelet
from methods.MTF_GLP import MTF_GLP
from methods.MTF_GLP_HPM import MTF_GLP_HPM
from methods.GSA import GSA
from methods.CNMF import CNMF
from methods.GFPCA import GFPCA
from metrics import ref_evaluate, no_ref_evaluate
from PIL import Image


list_ref = []
list_noref = []


def cal(ref, noref):
    reflist = []
    noreflist = []
    # print("ref[:][0]")
    # print([ii[0] for ii in ref])
    reflist.append(np.mean([ii[0] for ii in ref]))
    reflist.append(np.mean([ii[1] for ii in ref]))
    reflist.append(np.mean([ii[2] for ii in ref]))
    reflist.append(np.mean([ii[3] for ii in ref]))
    reflist.append(np.mean([ii[4] for ii in ref]))
    reflist.append(np.mean([ii[5] for ii in ref]))

    noreflist.append(np.mean([ih[0] for ih in noref]))
    noreflist.append(np.mean([ih[2] for ih in noref]))
    noreflist.append(np.mean([ih[2] for ih in noref]))
    return reflist, noreflist

path_ms = "/ghome/fuxy/yaogan/WV3_data/test128/ms"
path_pan = "/ghome/fuxy/yaogan/WV3_data/test128/pan"
#path_predict = "/ghome/fuxy/yaogan/WV3_data/test128/predict"
path_predict = "/ghome/fuxy/DPFN-master/results/pannet_WV3/test"

list_name = []
for file_path in os.listdir(path_ms):
    list_name.append(file_path)
print("name---------------")
print(list_name)
#num = len(list_name)
num=2
fnb = 0
for file_name_i in list_name:
    '''loading data'''
    fnb = fnb+1
    path_ms_file = os.path.join(path_ms, file_name_i)
    path_pan_file = os.path.join(path_pan, file_name_i)
    path_predict_file = os.path.join(path_predict, file_name_i)
    # print(path_ms_file,path_pan_file)

    original_msi = np.array(Image.open(path_ms_file))
    original_pan = np.array(Image.open(path_pan_file))
    fused_image = np.array(Image.open(path_predict_file))
    '''normalization'''
    # max_patch, min_patch = np.max(original_msi, axis=(0,1)), np.min(original_msi, axis=(0,1))
    print("-----eeeeeeeeeeeeeeeeeeeeeee")
    print(original_msi)
    print("-----eeeeeeeeeeeeeeeeeeeeeee")
    print(original_pan)
    print("-----eeeeeeeeeeeeeeeeeeeeeee")
    print(fused_image)
    print("-----eeeeeeeeeeeeeeeeeeeeeee")
    gt = original_msi
    # max_patch, min_patch = np.max(original_pan, axis=(0,1)), np.min(original_pan, axis=(0,1))
    fused_image = np.float32(fused_image)

    used_ms = cv2.resize(original_msi, (original_msi.shape[1]//4, original_msi.shape[0]//4), cv2.INTER_CUBIC)
    used_pan = np.expand_dims(original_pan, -1)

    print(used_ms)


    print('ms shape: ', used_ms.shape, 'pan shape: ', used_pan.shape)

    '''setting save parameters'''



    '''evaluating all methods'''
    ref_results={}
    ref_results.update({'metrics: ':'  PSNR,     SSIM,   SAM,    ERGAS,  SCC,    Q'})
    no_ref_results={}
    no_ref_results.update({'metrics: ':'  D_lamda, D_s,    QNR'})

    '''Bicubic method'''
    print('evaluating Bicubic method')

    print(gt.shape)
    temp_ref_results1 = ref_evaluate(fused_image, gt)
    temp_no_ref_results1 = no_ref_evaluate(fused_image, np.uint8(used_pan), np.uint8(used_ms))
    list_ref.append(temp_ref_results1)
    list_noref.append(temp_no_ref_results1)




    if fnb == num:

        print("------------------------------------------------ddddddd")
        print(list_ref)
        print(list_noref)
        temp_ref_results1, temp_no_ref_results1 = cal(list_ref, list_noref)
        ref_results.update({'deep   ':temp_ref_results1})
        no_ref_results.update({'deep    ':temp_no_ref_results1})


        print('################## reference comparision #######################')
        for index, i in enumerate(ref_results):
            if index == 0:
                print(i, ref_results[i])
            else:
                print(i, [round(j, 4) for j in ref_results[i]])
        print('################## reference comparision #######################')


        print('################## no reference comparision ####################')
        for index, i in enumerate(no_ref_results):
            if index == 0:
                print(i, no_ref_results[i])
            else:
                print(i, [round(j, 4) for j in no_ref_results[i]])
        print('################## no reference comparision ####################')

        break


