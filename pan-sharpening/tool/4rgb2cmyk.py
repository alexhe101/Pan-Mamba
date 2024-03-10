#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-12-03 20:27:24
LastEditTime: 2020-12-05 16:35:11
Description: 4通道RGB转换为CMYK颜色空间
'''
from PIL import Image
import os
import numpy as np

def rgb2cmyk(image_path, save_path):
    image = Image.open(image_path)
    np_image = np.array(image)
    copy = np_image.copy()
    copy[:, :, 0], copy[:, :, 2] = np_image[:, :, 2], np_image[:, :, 0]
    img = Image.fromarray(copy).convert('CMYK')
    img.save(save_path)

if __name__ == "__main__":
    rgb_path = '/Volumes/Elements/PAN_MS/uint8/GF1/MS/'
    cmyk_path = '/Volumes/Elements/PAN_MS/uint8/GF1/MS1/'
    for i in range(1,11):
        img_rgb = os.path.join(rgb_path, 'ms'+str(i)+'.tif')
        img_cmyk = os.path.join(cmyk_path, str(i)+'.tif')
        rgb2cmyk(img_rgb, img_cmyk)