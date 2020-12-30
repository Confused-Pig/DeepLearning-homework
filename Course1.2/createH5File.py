#!/user/bin/python3
# Author:Confused Pig
# -*- coding: utf-8 -*-

# @Time    : 2020/12/30  12:52
# @Author  : Confused Pig
# @Site    : 
# @File    : createH5File.py
# @Software: PyCharm

import os
import numpy as np
import cv2
import h5py

def ResizePic(path,save_path,dim):
    i = 1
    file_name = os.listdir(path)
    for img in file_name:
        image = cv2.imread(path + img)
        image_size = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_path + str(i) + '.jpg', image_size)
        i = i + 1


def save_image_to_h5py(path):
    img_list = []
    label_list = []

    for child_dir in os.listdir(path):

        img = cv2.imread(os.path.join(path,child_dir))
        img_list.append(img)
        label_list.append(1)

    img_np = np.array(img_list)

    label_np = np.array(label_list)
    print('数据集标签顺序：\n',label_np)

    #'a' ，如果已经有这个名字的h5文件存在将不会打开，目的为了防止误删信息。
    #‘w' ，如果有同名文件也能打开，但会覆盖上次的内容。
    with h5py.File('datasets/test_cat.h5','a') as f:
        f.create_dataset('test_cat',data = img_np)
        f.create_dataset('test_label',data = label_np)

        f.close()

path = 'Image/pic/'
save_path = 'Image/test_pic/'
dim = (64,64)

d = ResizePic(path,save_path,dim)
b = save_image_to_h5py(save_path)