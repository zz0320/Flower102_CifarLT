# -*- coding: utf-8 -*-
"""
# @file name    :parse_error_imgs.py
# @author       :zz0320
# @data         :2022-4-5
# @brief        :将flower数据集按照类别进行存放 便于后序分析
"""

import os
import pickle
import shutil

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)

if __name__ == '__main__':
    root_dir = r"/Users/kenton/Downloads/deeplearning_dataset/flower102"
    path_mat = r"/Users/kenton/Downloads/deeplearning_dataset/flower102/imagelabels.mat"
    reorder_dir = os.path.join(root_dir, "recorder")
    jpg_dir = os.path.join(root_dir,"jpg")

    from scipy.io import loadmat
    label_array = loadmat(path_mat)["labels"].squeeze()

    names = os.listdir(jpg_dir)
    names = [p for p in names if p.endswith(".jpg")]
    for name in names:
        idx = int(name[6:11])
        label = label_array[idx - 1] - 1
        out_dir = os.path.join(reorder_dir, str(label))
        path_src = os.path.join(jpg_dir, name)
        my_mkdir(out_dir)
        shutil.copy(path_src, out_dir) # 复制文件








