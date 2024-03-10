#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-06-13 20:12:23
LastEditTime: 2020-12-23 18:52:56
@Description: file content
'''
import numpy as np 
import glob, os, h5py
import os
import cv2
from scipy import misc
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

class image_to_patch:
    
    def __init__(self, patch_size, scale, ms_path, ms_image_path, pan_path, pan_image_path):

        self.stride = patch_size
        self.scale = scale
        self.ms_path = ms_path
        self.ms_image_path = ms_image_path
        self.pan_path = pan_path
        self.pan_image_path = pan_image_path
        
        if not os.path.exists(ms_image_path):
            os.mkdir(ms_image_path)
        if not os.path.exists(pan_image_path):
            os.mkdir(pan_image_path)

    def to_patch(self):
        
        #read MS image, CMYK space
        img_ms = self.imread(self.ms_path)
        img_ms = self.modcrop(img_ms, self.scale)
        
        #read PAN image, Gray space
        img_pan = self.imread(self.pan_path)
        img_pan = self.modcrop(img_pan, self.scale)
        
        # high, width
        h, w  = img_ms.size
        n_ms = 1
        n_pan = 1
        
        box = [0 , w-self.stride , 0 + self.stride,  w]
        sub_img_label = img_ms.crop(box)
        sub_img_label.save(os.path.join(self.ms_image_path, str(n_ms)+'.tif'))

        h, w  = img_pan.size
        box = [0 , w-self.stride *  self.scale, 0+ self.stride * self.scale,  w]
        sub_img_label = img_pan.crop(box)
        sub_img_label.save(os.path.join(self.pan_image_path, str(n_pan)+'.tif'))     


    def imread(self, path):
        img = Image.open(path)
        return img

    def modcrop(self, img, scale =3):
        h, w = img.size
        h = (h // scale) * scale
        w = (w // scale) * scale
        box=(0,0,h,w)
        img = img.crop(box)
        return img  
    
if __name__ == '__main__':
    ms_image_size = 256
    scale = 4
    ms_pach = r'/Users/wjmecho/Desktop/pan/1/8.tif'
    ms_image_path = r'./ms'
    pan_pach = r'/Users/wjmecho/Desktop/pan/1/8pan.tif'
    pan_image_path = r'./pan'

    # select image patch
    task = image_to_patch(ms_image_size, scale, ms_pach, ms_image_path, pan_pach, pan_image_path)
    task.to_patch()
