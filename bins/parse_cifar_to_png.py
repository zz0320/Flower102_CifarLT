# -*- coding: utf-8 -*-
"""
# @file name  : parse_cifar10_to_png.py
# @author     : zz0320
# @date       : 2021-04-11
# @brief      : 将cifar10数据pickle形式解析成png格式
"""
import numpy as np
import os
import sys
import pickle
from imageio import imwrite


def unpickle(file):
    fo = open(file, 'rb')
    if sys.version_info < (3, 0):
        dict_ = pickle.load(fo)
    else:
        dict_ = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict_


def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


def pasre_pickle_img(pkl_data):
    img = np.reshape(pkl_data[b'data'][i], (3, 32, 32))
    label_n = str(pkl_data[b'labels'][i])
    img = img.transpose((1, 2, 0))  # c*h*w --> h*w*c
    return img, label_n


def check_data_dir(path_data):
    if not os.path.exists(path_data):
        print("文件夹不存在，请检查数据是否存放到data_dir变量:{}".format(path_data))


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(__file__)
    cifar_dir = r"/Users/kenton/Downloads/deeplearning_dataset/cifar-10"    # 数据目录
    data_dir = os.path.join(cifar_dir, "cifar-10-batches-py")           # 源数据目录
    check_data_dir(data_dir)
    train_o_dir = os.path.join(cifar_dir, "cifar10_train")              # 输出的目录
    test_o_dir = os.path.join(cifar_dir, "cifar10_test")

    # train data
    for j in range(1, 6):
        data_path = os.path.join(data_dir,  "data_batch_" + str(j))  # data_batch_12345
        train_data = unpickle(data_path)
        print(data_path + " is loading...")

        for i in range(0, 10000):
            # 解析图片及标签
            img, label_num = pasre_pickle_img(train_data)
            # 创建文件夹
            o_dir = os.path.join(train_o_dir, label_num)
            my_mkdir(o_dir)
            # 保存图片
            img_name = label_num + '_' + str(i + (j - 1)*10000) + '.png'
            img_path = os.path.join(o_dir, img_name)
            imwrite(img_path, img)
        print(data_path + " loaded.")

    # test data
    test_data_path = os.path.join(data_dir, "test_batch")
    test_data = unpickle(test_data_path)
    for i in range(0, 10000):
        # 解析图片及标签
        img, label_num = pasre_pickle_img(test_data)
        # 创建文件夹
        o_dir = os.path.join(test_o_dir, label_num)
        my_mkdir(o_dir)
        # 保存图片
        img_name = label_num + '_' + str(i) + '.png'
        img_path = os.path.join(o_dir, img_name)
        imwrite(img_path, img)

    print("done.")
