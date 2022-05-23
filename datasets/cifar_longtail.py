# -*- coding: utf-8 -*-
"""
# @file name    :cifar_longtail.py
# @author       :zz0320
# @data         :2022-4-11
# @brief        :cifar-10长尾数据集的读取
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset


class CifarDataset(Dataset):
    names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    cls_num = len(names)

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []      # 定义list用于存储样本路径、标签
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label, path_img

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))   # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        for root, dirs, _ in os.walk(self.root_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.abspath(os.path.join(root, sub_dir, img_name))
                    label = int(sub_dir)
                    self.img_info.append((path_img, int(label)))
        random.shuffle(self.img_info)   # 将数据顺序打乱


class CifarLTDataset(CifarDataset):
    def __init__(self, root_dir, transform=None, imb_factor=0.01, isTrain=True):
        """
        :param root_dir:
        :param transform:
        :param imb_type:
        :param imb_factor: float, 值越小，数量下降越快,0.1表示最少的类是最多的类的0.1倍，如500：5000
        :param isTrain:
        """
        super(CifarLTDataset, self).__init__(root_dir, transform=transform)
        self.imb_factor = imb_factor
        if isTrain:
            self.nums_per_cls = self._get_img_num_per_cls()     # 计算每个类的样本数
            self._select_img()      # 采样获得符合长尾分布的数据量
        else:
            # 非训练状态，可采用均衡数据集测试
            self.nums_per_cls = []
            for n in range(self.cls_num):
                label_list = [label for p, label in self.img_info]  # 获取每个标签
                self.nums_per_cls.append(label_list.count(n))       # 统计每个类别数量

    def _select_img(self):
        """
        根据每个类需要的样本数进行挑选
        :return:
        """
        new_lst = []
        for n, img_num in enumerate(self.nums_per_cls):
            lst_tmp = [info for info in self.img_info if info[1] == n]  # 获取第n类别数据信息
            random.shuffle(lst_tmp)
            lst_tmp = lst_tmp[:img_num]
            new_lst.extend(lst_tmp)
        random.shuffle(new_lst)
        self.img_info = new_lst

    def _get_img_num_per_cls(self):
        """
        依长尾分布计算每个类别应有多少张样本
        :return:
        """
        img_max = len(self.img_info) / self.cls_num
        img_num_per_cls = []
        for cls_idx in range(self.cls_num):
            num = img_max * (self.imb_factor ** (cls_idx / (self.cls_num - 1.0)))  # 列出公式就知道了
            img_num_per_cls.append(int(num))
        return img_num_per_cls


if __name__ == "__main__":
    root_dir = r"/Users/kenton/Downloads/deeplearning_dataset/cifar-10/cifar10_train"
    train_dataset = CifarLTDataset(root_dir, imb_factor=0.01)
    print(len(train_dataset))
    print(next(iter(train_dataset)))
    print(train_dataset.nums_per_cls)
    #
    y = train_dataset.nums_per_cls
    x = range(len(y))
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    # plt.show()





