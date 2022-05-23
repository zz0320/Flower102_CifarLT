# -*- coding: utf-8 -*-
"""
# @file name    :flower102.py
# @author       :zz0320
# @data         :2022-3-30
# @brief        :DataSets类 数据读取
"""

import os
from PIL import Image
from torch.utils.data import Dataset

class FlowerDataset(Dataset):
    cls_num = 102 # 类的一些属性
    names = tuple([i for i in range(cls_num)])

    def __init__(self, root_dir, transform=None): #定义核心变量 比如路径 和 Transform预处理
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = [] # [(path, label),(),...]
        self.label_array = None

        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index，从硬盘中读取数据，并预处理 to tensor
        :param index:
        :return:
        """
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label, path_img

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir!".format(self.root_dir))
        return len(self.img_info)

    def _get_img_info(self): # 读取数据的路径和标签 存在一个列表当中 给__getitem__使用
        """
        实现数据集的读取，将硬盘中的数据路径和标签 读取进来 存在一个list中
        path, label
        :return:
        """
        names_imgs = os.listdir(self.root_dir)
        names_imgs = [n for n in names_imgs if n.endswith('.jpg')] # pythonic 列表推导式

        # 读取mat形式label
        label_file = "imagelabels.mat"
        path_label_file = os.path.join(self.root_dir,"..",label_file)
        from scipy.io import loadmat
        label_array = loadmat(path_label_file)["labels"].squeeze()
        self.label_array = label_array

        # 匹配label
        idx_imgs = [int(n[6:11]) for n in names_imgs] # 标号
        path_imgs = [os.path.join(self.root_dir, n) for n in names_imgs] #  路径 通过名称拼上根目录 列表推导式
        self.img_info = [(p, int(label_array[idx - 1] - 1)) for p, idx in zip(path_imgs, idx_imgs)]
        # 获取整个图片的info 包含 path 和 label

# test part
if __name__ == "__main__":
    root_dir = r'/Users/kenton/Downloads/deeplearning_dataset/flower102/train'
    test_dataset = FlowerDataset(root_dir)
    print(len(test_dataset))

    print(next(iter(test_dataset))) #迭代器



