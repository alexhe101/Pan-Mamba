#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-12-03 09:21:22
LastEditTime: 2020-12-23 10:41:59
Description: 去除图像中无像素区域，避免无法计算梯度
'''
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

class modcrop:
    """
    docstring
    """
    def __init__(self, pan_path, ms_path, pixel):

        self.pan_path = pan_path
        self.ms_path = ms_path
        self.pixel = pixel
        self.pan_img = self.read(pan_path) 
        self.ms_img = self.read(ms_path) 
        self.run(self.pan_img, self.ms_img, self.pixel)

    def read(self, path):
        img = Image.open(path)
        return img
        
    def save(self, img, patch):
        """
        docstringls
        """
        img.save(path)
    
    def run(self, pan_img, ms_img, pixel):

        # if pan_img.size != ms_img.size:
        #     raise AssertionError('Error size!')
        box1 = (pixel, pixel, ms_img.size[0] - pixel, ms_img.size[1] - pixel)
        box = (4*pixel, 4*pixel, pan_img.size[0] - 4*pixel, pan_img.size[1] - 4*pixel)
        ms_img.crop(box1).save('/ghome/fuxy/WV3/MS.tif')
        pan_img.crop(box).save('/ghome/fuxy/WV3/PAN.tif')

if __name__ == "__main__":

    #ms_path = '/Users/wjmecho/Desktop/pan/8.tif'
    #pan_path = '/Users/wjmecho/Desktop/pan/8.tif'
    ms_path = "/ghome/fuxy/WV3/ms.tif"
    pan_path = "/ghome/fuxy/WV3/pan.tif"
    app = modcrop(pan_path, ms_path, 64)