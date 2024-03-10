import torch

import numpy as np
import os
import cv2

# from skimage.metrics import peak_signal_noise_ratio as psnr

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# def get_edge(data):
#     data = data.cpu().numpy()
#     rs = np.zeros_like(data)
#     N = data.shape[0]  # batch_size
#     for i in range(N):
#         # if len(data.shape)==3:
#         #     rs[i,:,:] = data[i,:,:] - cv2.boxFilter(data[i,:,:],-1,(5,5))  # 针对pan 因为这个图片只有一个通道
#         # else:
#         rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))  # 针对 ms
#     rs = torch.tensor(rs).to(device)
#     return rs

def get_edge(data):
    # 通道转换，先从 NCHW 变成 NHWC
    data = data.permute(0, 2, 3, 1)
    data = data.cpu().numpy()
    rs = np.zeros_like(data)
    N = data.shape[0]  # batch_size
    for i in range(N):
        # if PAN:
        if data.shape[3]==1:
            temp1 = cv2.boxFilter(data[0, :, :, :], -1, (5, 5))
            temp1 = torch.from_numpy(temp1).unsqueeze(0).permute(1,2,0)
            temp1 = temp1.numpy()
            rs[i, :, :, :] = data[i, :, :, :] - temp1
        # if MS:
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))  # 针对 ms
    rs = torch.tensor(rs).to(device)
    # 再转回 NCHW
    rs = rs.permute(0, 3, 1, 2)
    return rs

def create_grid(resolution):
    grid_y, grid_x = torch.meshgrid([torch.linspace(-1, 1, steps=resolution[1]),
                                     torch.linspace(-1, 1, steps=resolution[0])])
    grid = torch.stack([grid_y, grid_x], dim=-1).to(device).requires_grad_()

    return grid


def get_b(spectral_num, idx):
    b_start = -1
    b_end = 1
    b = (b_end - b_start) / (spectral_num - 1) * idx
    b -= 1

    return b


# def cal_psnr(img1, img2):

#     return psnr(img1, img2)

import matplotlib.pyplot as plt
import math
def show_feature_map(feature_map,layer,name='rgb',rgb=False):
    feature_map = feature_map.squeeze(0)
    #if rgb: feature_map = feature_map.permute(1,2,0)*0.5+0.5
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = math.ceil(np.sqrt(feature_map_num))
    if rgb:
        #plt.figure()
        #plt.imshow(feature_map)
        #plt.axis('off')
        feature_map = cv2.cvtColor(feature_map,cv2.COLOR_BGR2RGB)
        cv2.imwrite('data/'+layer+'/'+name+".png",feature_map*255)
        #plt.show()
    else:
        plt.figure()
        for index in range(1, feature_map_num+1):
            t = (feature_map[index-1]*255).astype(np.uint8)
            t = cv2.applyColorMap(t, cv2.COLORMAP_TWILIGHT)
            plt.subplot(row_num, row_num, index)
            plt.imshow(t, cmap='gray')
            plt.axis('off')
            #ensure_path('data/'+layer)
            cv2.imwrite('data/'+layer+'/'+str(name)+'_'+str(index)+".png",t)
        #plt.show()
        plt.savefig('data/'+layer+'/'+str(name)+".png")
        
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret.to(device)

def make_coord_sro(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    #ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

if __name__ == '__main__':

    for i in range(8):
        pred_path = os.path.join('output/v6/pred/step_36000/pred/batch_00/img_00/', '%d.png' % i)
        gt_path = os.path.join('output/v6/pred/step_36000/gt/batch_00/img_00/', '%d.png' % i)
        pred = cv2.imread(pred_path, -1)
        gt = cv2.imread(gt_path, -1)

        print(cal_psnr(pred, gt))