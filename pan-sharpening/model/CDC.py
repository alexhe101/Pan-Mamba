import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np


class cdc_vg(nn.Module):
    def __init__(self, mid_ch, theta=0.7):

        super(cdc_vg, self).__init__()

        self.cdc = Conv2d_cd(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False, theta= theta)
        self.cdc_bn = nn.BatchNorm2d(mid_ch)
        self.cdc_act = nn.PReLU()

        self.h_conv = Conv2d_Hori_Veri_Cross(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.d_conv = Conv2d_Diag_Cross(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.vg_bn = nn.BatchNorm2d(mid_ch)
        self.vg_act = nn.PReLU()

        # self.HP_branch = Parameter(torch.FloatTensor(1))

    def forward(self, x):
        out_0 = self.cdc_act(self.cdc_bn(self.cdc(x)))

        out1 = self.h_conv(out_0)
        out2 = self.d_conv(out_0)
        out = self.vg_act(self.vg_bn(0.5 * out1 + 0.5 * out2))
        # out = out1 + out2
        return out + x


class ResBlock_cdc(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1, theta=0.8):

        super(ResBlock_cdc, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

        self.h_conv = Conv2d_Hori_Veri_Cross(in_channels=n_feats, out_channels=n_feats, kernel_size=3,
                                             stride=1, padding=1, bias=False, theta=theta)
        self.d_conv = Conv2d_Diag_Cross(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1,
                                        padding=1, bias=False, theta=theta)
        # self.HP_branch = Parameter(torch.FloatTensor(1))

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        out1 = self.h_conv(x)
        out2 = self.d_conv(x)
        # out = torch.sigmoid(self.HP_branch) * out1 + (1 - torch.sigmoid(self.HP_branch)) * out2
        out = out1 + out2

        res += x + out

        return res


class cdcconv(nn.Module):
    def __init__(self, in_channels, out_channels, theta=0.8):

        super(cdcconv, self).__init__()

        self.h_conv = Conv2d_Hori_Veri_Cross(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.d_conv = Conv2d_Diag_Cross(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)

        self.HP_branch = Parameter(torch.FloatTensor(1))

    def forward(self, x):
        out1 = self.h_conv(x)
        out2 = self.d_conv(x)
        out = torch.sigmoid(self.HP_branch) * out1 + (1 - torch.sigmoid(self.HP_branch)) * out2 + x
        # out = out1 + out2
        return out


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff



class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Hori_Veri_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1],
                                 self.conv.weight[:, :, :, 2], self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class Conv2d_Diag_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Diag_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1], tensor_zeros,
                                 self.conv.weight[:, :, :, 2], tensor_zeros, self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

