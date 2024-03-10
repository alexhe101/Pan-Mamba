# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------------

import os
import json

def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
        
def save_param(input_dict, path):
    f = open(path, 'w')
    f.write(json.dumps(input_dict))
    f.close()
    print("Hyper-Parameters have been saved!")
    

# ----------------------------------------------------------------------------
# Dataset & Image Processing
# ----------------------------------------------------------------------------

import os
import h5py
import torch

from glob import glob
import numpy as np
import torch.utils.data as Data

from scipy.io import loadmat


def normlization(x):
    # x [N,C,H,W]
    N,C,H,W = x.shape
    m = []
    for i in range(N):
        m.append(torch.max(x[i,:,:,:]))
    m = torch.stack(m, dim=0)[:,None,None,None]
    m = m+1e-10
    x = x/m
    return x,m

def inverse_normlization(x, m):
    return x*m
    
def im2double(img):
    if img.dtype=='uint8':
        img = img.astype(np.float32)/255.
    elif img.dtype=='uint16':
        img = img.astype(np.float32)/65535.
    else:
        img = img.astype(np.float32)
    return img

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def imresize(img, size=None, scale_factor=None):
    # img (np.array) - [C,H,W]
    imgT = torch.from_numpy(img).unsqueeze(0) #[1,C,H,W]
    if size is None and scale_factor is not None:
        imgT = torch.nn.functional.interpolate(imgT, scale_factor=scale_factor)
    elif size is not None and scale_factor is None:
        imgT = torch.nn.functional.interpolate(imgT, size=size)
    else:
        print('Neither size nor scale_factor is given.')
    imgT = imgT.squeeze(0).numpy()
    return imgT
    
  
def prepare_data(data_path, 
                 patch_size, 
                 aug_times=4,
                 stride=25, 
                 synthetic=True, 
                 scale=2,
                 file_name='train.h5'
                 ):
    # patch_size : the window size of low-resolution images
    # scale : the spatial ratio between low-resolution and guide images
    # train
    print('process training data')
    files = glob(os.path.join(data_path, 'train', '*.mat'))
    h5f = h5py.File(file_name, 'w')
    h5gt = h5f.create_group('GT')
    h5guide = h5f.create_group('PAN')
    h5lr = h5f.create_group('MS')
    train_num = 0
    for i in range(len(files)):
        img = loadmat(files[i])
        lr = img['I_MS'].astype('float32') # [Height, Width, Channels]
        guide = img['I_PAN'].astype('float32') # [Height, Width]
        # print([lr.shape,guide.shape])
        lr = np.transpose(lr, [2,0,1]) # [Channels, Height, Width]
        guide = guide[None,:,:]  # [1, Height, Width]
        
        if synthetic:
            # if synthetic is True: the spatial resolutions of lr and guide are the same
            lr_patches = Im2Patch(lr, win=scale*patch_size, stride=stride) #[C,H,W,N]
            guide_patches = Im2Patch(guide, win=scale*patch_size, stride=stride)
        else:
            scale = int(guide.shape[-1]/lr.shape[-1])
            # print(scale)
            guide = imresize(guide, size=lr.shape[1:])
            lr_patches = Im2Patch(lr, win=scale*patch_size, stride=stride) #[C,H,W,N]
            guide_patches = Im2Patch(guide, win=scale*patch_size, stride=stride)
            
        print("file: %s # samples: %d" % (files[i], lr_patches.shape[3]*aug_times))
        for n in range(lr_patches.shape[3]):
            gt_data = lr_patches[:,:,:,n].copy()
            guide_data = guide_patches[:,:,:,n].copy()
            lr_data = imresize(gt_data, scale_factor=1/scale)
            
            h5gt.create_dataset(str(train_num), 
                                data=gt_data, dtype=gt_data.dtype,shape=gt_data.shape)
            h5guide.create_dataset(str(train_num), 
                                   data=guide_data, dtype=guide_data.dtype,shape=guide_data.shape)
            h5lr.create_dataset(str(train_num), 
                                data=lr_data, dtype=lr_data.dtype,shape=lr_data.shape)
            train_num += 1
            for m in range(aug_times-1):
                gt_data_aug = np.rot90(gt_data, m+1, axes=(1,2))
                guide_data_aug = np.rot90(guide_data, m+1, axes=(1,2))
                lr_data_aug = np.rot90(lr_data, m+1, axes=(1,2))
                
                h5gt.create_dataset(str(train_num)+"_aug_%d" % (m+1), 
                                    data=gt_data_aug, dtype=gt_data_aug.dtype,shape=gt_data_aug.shape)
                h5guide.create_dataset(str(train_num)+"_aug_%d" % (m+1), 
                                       data=guide_data_aug, dtype=guide_data_aug.dtype,shape=guide_data_aug.shape)
                h5lr.create_dataset(str(train_num)+"_aug_%d" % (m+1), 
                                    data=lr_data_aug, dtype=lr_data_aug.dtype,shape=lr_data_aug.shape)
                train_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)

class CaveH5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['Guide'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        guide = np.array(h5f['Guide'][key])
        gt    = np.array(h5f['GT'][key])
        lr    = np.array(h5f['LR'][key])
        h5f.close()
        return torch.Tensor(lr),torch.Tensor(guide),torch.Tensor(gt)

class CaveDataset(Data.Dataset):
    def __init__(self, root, scale):
        self.scale = scale
        self.root = root
        self.files = glob(root+'/*.mat')

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        temp  = h5py.File(self.files[index])
        guide = im2double(temp['Guide'][:])
        gt    = im2double(temp['LR'][:])
        lr    = imresize(gt, scale_factor=1/self.scale)
        del temp
        return lr, guide, gt

class PSDataset(Data.Dataset):
    def __init__(self, root, scale):
        self.scale = scale
        self.root = root
        self.files = glob(root+'/*.mat')

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        temp  = loadmat(self.files[index])
        gt    = np.transpose(temp['I_MS'].astype('float32'), [2,0,1])
        pan   = temp['I_PAN'].astype('float32')[None,:,:]
        ms    = imresize(gt, scale_factor=1/self.scale)
        pan   = imresize(pan, scale_factor=1/self.scale)
        del temp
        return ms, pan, gt

class PSH5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['PAN'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        pan = np.array(h5f['PAN'][key])
        gt = np.array(h5f['GT'][key])
        ms = np.array(h5f['MS'][key])
        h5f.close()
        return torch.Tensor(ms),torch.Tensor(pan),torch.Tensor(gt)


class PSH5Datasetfu(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(self.h5file_path, 'r')
        gt = h5f['gt'][...]  ## ground truth N*H*W*C
        pan = h5f['pan'][...]  #### Pan image N*H*W
        ms = h5f['ms'][...]  ### low resolution MS image
        self.N = gt.shape[0]

        self.gt = np.array(gt, dtype=np.float32) / 2047.  ### normalization, WorldView L = 11
        self.pan = np.array(pan, dtype=np.float32) / 2047.
        self.ms = np.array(ms, dtype=np.float32) / 2047.

        h5f.close()

    def __len__(self):
        return self.N

    def __getitem__(self, index):

        train_gt = self.gt[index, :, :, :]
        #train_gt = train_gt[np.newaxis,:,:,:]
        train_gt = np.transpose(train_gt, (2, 0, 1))

        train_pan = self.pan[index, :, :]

        train_pan = train_pan[:, :, np.newaxis] # expand to N*H*W*1; new added!
        train_pan = np.transpose(train_pan, (2, 0, 1))

        train_ms = self.ms[index, :, :, :]
        #train_ms = train_ms[np.newaxis, :, :, :]
        train_ms = np.transpose(train_ms, (2, 0, 1))


        return torch.Tensor(train_ms), torch.Tensor(train_pan), torch.Tensor(train_gt)


# ----------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=2, pool_types=['avg', 'max'], no_spatial=False, no_channel=True):
        super(CBAM, self).__init__()
        self.no_channel = no_channel
        if not no_channel:
            self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        if not self.no_channel:
            x = self.ChannelGate(x)
        if not self.no_spatial:
            x = self.SpatialGate(x)
        return x

# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn

eps = torch.finfo(torch.float32).eps

# RAP loss
class RAP(nn.Module):
    def __init__(self, lap_weight=1, angle_weight=1):
        super(RAP, self).__init__()
        self.lap_weight = lap_weight
        self.angle_weight = angle_weight
    
    def forward(self, img, gt):
        return nn.L1Loss()(img, gt) + self.lap_weight * lap_loss(img,gt) + self.angle_weight * sam(img, gt)
        
def lap_loss(img, gt):
    img = laplacian(img, 3)
    gt  = laplacian(gt, 3)
    return nn.L1Loss()(img, gt)

def rmse(img, gt):
    """RMSE for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N,C,_,_ = img.shape
    img = torch.reshape(img, [N,C,-1])
    gt  = torch.reshape(gt,  [N,C,-1])
    mse = (img-gt).pow(2).sum(dim=-1)
    rmse = mse/(gt.pow(2).sum(dim=-1)+eps)
    rmse = rmse.mean(dim=-1).sqrt()
    return rmse.mean()
    
def sam(img1, img2):
    """SAM for (N, C, H, W) image; torch.float32 [0.,1.]."""
    inner_product = (img1 * img2).sum(dim=1)
    img1_spectral_norm = torch.sqrt((img1**2).sum(dim=1))
    img2_spectral_norm = torch.sqrt((img2**2).sum(dim=1))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + eps)).clamp(min=0, max=1)
    cos_theta = cos_theta.reshape(cos_theta.shape[0], -1)
    return torch.mean(torch.acos(cos_theta), dim=-1).mean()

# ----------------------------------------------------------------------------
# Guided Filter
# ----------------------------------------------------------------------------
'''
This code is written by Huikai Wu. Original code is available at
https://github.com/wuhuikai/DeepGuidedFilter/tree/master/GuidedFilteringLayer/GuidedFilter_PyTorch

Please cite:
Fast End-to-End Trainable Guided Filter
Huikai Wu, Shuai Zheng, Junge Zhang, Kaiqi Huang
CVPR 2018
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)

class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, lr_x, lr_y, hr_x):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        ## mean_x
        mean_x = self.boxfilter(lr_x) / N
        ## mean_y
        mean_y = self.boxfilter(lr_y) / N
        ## cov_xy
        cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
        ## var_x
        var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x

        ## A
        A = cov_xy / (var_x + self.eps)
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A*hr_x+mean_b


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b





# ----------------------------------------------------------------------------
# Kornia
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from typing import Tuple, List

def compute_padding(kernel_size: Tuple[int, int]) -> List[int]:
    """Computes padding tuple."""
    # 4 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) == 2, kernel_size
    computed = [(k - 1) // 2 for k in kernel_size]
    return [computed[1], computed[1], computed[0], computed[0]]


def filter2D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel type is not a torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.to(input.device).to(input.dtype)
    tmp_kernel = tmp_kernel.repeat(c, 1, 1, 1)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # convolve the tensor with the kernel
    return F.conv2d(input_pad, tmp_kernel, padding=0, stride=1, groups=c)

def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))

def get_box_kernel2d(kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Utility function that returns a box filter."""
    kx: float = float(kernel_size[0])
    ky: float = float(kernel_size[1])
    scale: torch.Tensor = torch.tensor(1.) / torch.tensor([kx * ky])
    tmp_kernel: torch.Tensor = torch.ones(1, kernel_size[0], kernel_size[1])
    return scale.to(tmp_kernel.dtype) * tmp_kernel

class BoxBlur(nn.Module):
    r"""Blurs an image using the box filter.

    The function smooths an image using the kernel:

    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): if True, L1 norm of the kernel is set to 1.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = kornia.filters.BoxBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 border_type: str = 'reflect',
                 normalized: bool = True) -> None:
        super(BoxBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.border_type: str = border_type
        self.kernel: torch.Tensor = get_box_kernel2d(kernel_size)
        self.normalized: bool = normalized
        if self.normalized:
            self.kernel = normalize_kernel2d(self.kernel)

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(kernel_size=' + str(self.kernel_size) + ', ' +\
            'normalized=' + str(self.normalized) + ', ' + \
            'border_type=' + self.border_type + ')'

    def forward(self, input: torch.Tensor):  # type: ignore
        return filter2D(input, self.kernel, self.border_type)

# functiona api
def box_blur(input: torch.Tensor,
             kernel_size: Tuple[int, int],
             border_type: str = 'reflect',
             normalized: bool = True) -> torch.Tensor:
    r"""Blurs an image using the box filter.

    See :class:`~kornia.filters.BoxBlur` for details.
    """
    return BoxBlur(kernel_size, border_type, normalized)(input)

class PSNRLoss(nn.Module):
    r"""Creates a criterion that calculates the PSNR between 2 images. Given an m x n image,
    .. math::
    \text{MSE}(I,T) = \frac{1}{m\,n}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1} [I(i,j) - T(i,j)]^2

    Arguments:
        max_val (float): Maximum value of input

    Shape:
        - input: :math:`(*)`
        - approximation: :math:`(*)` same shape as input
        - output: :math:`()` a scalar

    Examples:
        >>> kornia.losses.psnr(torch.ones(1), 1.2*torch.ones(1), 2)
        tensor(20.0000) # 10 * log(4/((1.2-1)**2)) / log(10)

    reference:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    """

    def __init__(self, max_val: float) -> None:
        super(PSNRLoss, self).__init__()
        self.max_val: float = max_val

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore
        return psnr_loss(input, target, self.max_val)


def psnr_loss(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    r"""Function that computes PSNR

    See :class:`~kornia.losses.PSNR` for details.
    """
    if not torch.is_tensor(input) or not torch.is_tensor(target):
        raise TypeError(f"Expected 2 torch tensors but got {type(input)} and {type(target)}")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")
    mse_val = mse_loss(input, target, reduction='mean')
    max_val_tensor: torch.Tensor = torch.tensor(max_val).to(input.device).to(input.dtype)
    return 10 * torch.log10(max_val_tensor * max_val_tensor / mse_val)


def _compute_zero_padding(kernel_size: int) -> int:
    """Computes zero padding."""
    return (kernel_size - 1) // 2

def gaussian(window_size, sigma):
    x = torch.arange(window_size).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    return gauss / gauss.sum()

def get_gaussian_kernel1d(kernel_size: int,
                          sigma: float,
                          force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples::

        >>> kornia.image.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> kornia.image.get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d

def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    return kernel_2d

class SSIM(nn.Module):
    r"""Creates a criterion that measures the Structural Similarity (SSIM)
    index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    the loss, or the Structural dissimilarity (DSSIM) can be finally described
    as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    Arguments:
        window_size (int): the size of the kernel.
        max_val (float): the dynamic range of the images. Default: 1.
        reduction (str, optional): Specifies the reduction to apply to the
         output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
         'mean': the sum of the output will be divided by the number of elements
         in the output, 'sum': the output will be summed. Default: 'none'.

    Returns:
        Tensor: the ssim index.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Target :math:`(B, C, H, W)`
        - Output: scale, if reduction is 'none', then :math:`(B, C, H, W)`

    Examples::

        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> ssim = kornia.losses.SSIM(5, reduction='none')
        >>> loss = ssim(input1, input2)  # 1x4x5x5
    """

    def __init__(
            self,
            window_size: int,
            reduction: str = "none",
            max_val: float = 1.0) -> None:
        super(SSIM, self).__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.reduction: str = reduction

        self.window: torch.Tensor = get_gaussian_kernel2d(
            (window_size, window_size), (1.5, 1.5))
        self.window = self.window.requires_grad_(False)  # need to disable gradients

        self.padding: int = _compute_zero_padding(window_size)

        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2

    def forward(  # type: ignore
            self,
            img1: torch.Tensor,
            img2: torch.Tensor) -> torch.Tensor:

        if not torch.is_tensor(img1):
            raise TypeError("Input img1 type is not a torch.Tensor. Got {}"
                            .format(type(img1)))

        if not torch.is_tensor(img2):
            raise TypeError("Input img2 type is not a torch.Tensor. Got {}"
                            .format(type(img2)))

        if not len(img1.shape) == 4:
            raise ValueError("Invalid img1 shape, we expect BxCxHxW. Got: {}"
                             .format(img1.shape))

        if not len(img2.shape) == 4:
            raise ValueError("Invalid img2 shape, we expect BxCxHxW. Got: {}"
                             .format(img2.shape))

        if not img1.shape == img2.shape:
            raise ValueError("img1 and img2 shapes must be the same. Got: {}"
                             .format(img1.shape, img2.shape))

        if not img1.device == img2.device:
            raise ValueError("img1 and img2 must be in the same device. Got: {}"
                             .format(img1.device, img2.device))

        if not img1.dtype == img2.dtype:
            raise ValueError("img1 and img2 must be in the same dtype. Got: {}"
                             .format(img1.dtype, img2.dtype))

        # prepare kernel
        b, c, h, w = img1.shape
        tmp_kernel: torch.Tensor = self.window.to(img1.device).to(img1.dtype)
        tmp_kernel = torch.unsqueeze(tmp_kernel, dim=0)

        # compute local mean per channel
        mu1: torch.Tensor = filter2D(img1, tmp_kernel)
        mu2: torch.Tensor = filter2D(img2, tmp_kernel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # compute local sigma per channel
        sigma1_sq = filter2D(img1 * img1, tmp_kernel) - mu1_sq
        sigma2_sq = filter2D(img2 * img2, tmp_kernel) - mu2_sq
        sigma12 = filter2D(img1 * img2, tmp_kernel) - mu1_mu2

        ssim_map = ((2. * mu1_mu2 + self.C1) * (2. * sigma12 + self.C2)) / \
            ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        loss = torch.clamp(ssim_map, min=0, max=1)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass
        return loss

def ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int,
        reduction: str = "mean",
        max_val: float = 1.0) -> torch.Tensor:
    r"""Function that measures the Structural Similarity (SSIM) index between
    each element in the input `x` and target `y`.

    See :class:`~kornia.losses.SSIM` for details.
    """
    return SSIM(window_size, reduction, max_val)(img1, img2)

# from typing import Tuple

def get_laplacian_kernel2d(kernel_size: int) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (int): filter size should be odd.

    Returns:
        Tensor: 2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])

        >>> kornia.image.get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])

    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(kernel_size))

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d

class Laplacian(nn.Module):
    r"""Creates an operator that returns a tensor using a Laplacian filter.

    The operator smooths the given tensor with a laplacian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        kernel_size (int): the size of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): if True, L1 norm of the kernel is set to 1.

    Returns:
        Tensor: the tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> laplace = kornia.filters.Laplacian(5)
        >>> output = laplace(input)  # 2x4x5x5
    """

    def __init__(self,
                 kernel_size: int, border_type: str = 'reflect',
                 normalized: bool = True) -> None:
        super(Laplacian, self).__init__()
        self.kernel_size: int = kernel_size
        self.border_type: str = border_type
        self.normalized: bool = normalized
        self.kernel: torch.Tensor = torch.unsqueeze(
            get_laplacian_kernel2d(kernel_size), dim=0)
        if self.normalized:
            self.kernel = normalize_kernel2d(self.kernel)

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(kernel_size=' + str(self.kernel_size) + ', ' +\
            'normalized=' + str(self.normalized) + ', ' + \
            'border_type=' + self.border_type + ')'

    def forward(self, input: torch.Tensor):  # type: ignore
        return filter2D(input, self.kernel, self.border_type)

def laplacian(
        input: torch.Tensor,
        kernel_size: int,
        border_type: str = 'reflect',
        normalized: bool = True) -> torch.Tensor:
    r"""Function that returns a tensor using a Laplacian filter.

    See :class:`~kornia.filters.Laplacian` for details.
    """
    return Laplacian(kernel_size, border_type, normalized)(input)