#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-12-03 10:34:18
LastEditTime: 2020-12-05 15:21:21
Description: file content
'''
import random, shutil, os

def random_select_test(test_num=78):
    
    root = r'/Users/wjmecho/Desktop/Non-pan/code/tool/ms'
    L1 = random.sample(range(1, 728), test_num+1)

    for i in range(1, int(test_num)+1):
        name = str(L1[i])+'.tif'
        newname=str(i)+'.tif'
        shutil.move(os.path.join('/Users/wjmecho/Desktop/Non-pan/code/tool/ms',name),os.path.join('/Users/wjmecho/Desktop/Non-pan/code/tool/test/ms',newname))
        shutil.move(os.path.join('/Users/wjmecho/Desktop/Non-pan/code/tool/pan',name),os.path.join('/Users/wjmecho/Desktop/Non-pan/code/tool/test/pan',newname))

if __name__ == "__main__":
    random_select_test()