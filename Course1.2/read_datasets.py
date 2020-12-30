#!/user/bin/python3
# Author:Confused Pig
# -*- coding: utf-8 -*-

# @Time    : 2020/12/29  16:58
# @Author  : Confused Pig
# @Site    : 
# @File    : read_datasets.py
# @Software: PyCharm

import numpy as np
import h5py

#Look at what things in the two h5 files
# with h5py.File('datasets/test_catvnoncat.h5','r') as f:
#     for keys in f.keys():
#         print(f[keys],keys,f[keys].name)
#     print(f['test_set_y'][:])
#     print(f['list_classes'][:])
#     f.close()

'''
    we can see: it keeps datas like dictionary
        <HDF5 dataset "list_classes": shape (2,), type "|S7"> list_classes /list_classes
        <HDF5 dataset "test_set_x": shape (50, 64, 64, 3), type "|u1"> test_set_x /test_set_x
        <HDF5 dataset "test_set_y": shape (50,), type "<i8"> test_set_y /test_set_y
        [1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 0 0 1 1 1 1 0 1 0 1 1 1 1 0 0 0 1 0 0 1 1 1 0 0 0 1 1 1 0]
        [b'non-cat' b'cat']

    list_classes:cat or not
    test_set_x:50 pictures (64*64*3)
    test_set_y:labels
    
'''

#load datas
def load_datasets():
    test_datas = h5py.File('datasets/test_catvnoncat.h5','r')
    test_set_x_orig = np.array(test_datas['test_set_x'][:])
    test_set_y_orig = np.array(test_datas['test_set_y'][:])

    train_datas = h5py.File('datasets/train_catvnoncat.h5', 'r')
    train_set_x_orig = np.array(train_datas['train_set_x'][:])
    train_set_y_orig = np.array(train_datas['train_set_y'][:])

    classes = np.array(test_datas['list_classes'][:])

    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))

    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes