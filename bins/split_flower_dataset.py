# -*- coding: utf-8 -*-
"""
# @file name    :split_flower_dataset.py
# @author       :zz0320
# @data         :2022-3-30
# @brief        :划分flower数据集
"""

import os
import random
import shutil

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)

def move_img(imgs, root_dir, setname):
    data_dir = os.path.join(root_dir, setname)
    my_mkdir(data_dir)
    for idx, path_imgs in enumerate(imgs):
        print("{} / {}".format(idx, len(imgs)))
        shutil.copy(path_imgs, data_dir) # 关键是这一行代码 移动图片到新文件夹
    print("{} dataset, copy {} img to {}".format(setname, len(imgs), data_dir))

if __name__ == '__main__':
    # 0. config
    random_seed = 20210309
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    root_dir = r'/Users/kenton/Downloads/deeplearning_dataset/flower102'

    # 1. read list, shuffle
    data_dir = os.path.join(root_dir, "jpg")
    name_imgs = [p for p in os.listdir(data_dir) if p.endswith(".jpg")] # 列表生成式
    path_imgs = [os.path.join(data_dir, name) for name in name_imgs]
    random.seed(random_seed)
    random.shuffle(path_imgs)
    print(path_imgs[0])

    # 2. split
    train_breakpoints = int(len(path_imgs) * train_ratio)
    valid_breakpoints = int(len(path_imgs) * (train_ratio + valid_ratio))
    test_breakpoints = int(len(path_imgs) * 1) # 可以省略

    train_img = path_imgs[: train_breakpoints]
    valid_img = path_imgs[train_breakpoints: valid_breakpoints]
    test_img = path_imgs[valid_breakpoints:]

    # 3. copy and save
    move_img(train_img, root_dir, "train")
    move_img(valid_img, root_dir, "valid")
    move_img(test_img, root_dir, "test")


