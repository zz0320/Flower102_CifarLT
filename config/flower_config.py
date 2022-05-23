# -*- coding: utf-8 -*-
"""
# @file name    :flower_config.py
# @author       :zz0320
# @data         :2022-4-5
# @brief        :花朵分类网络参数配置
"""

import argparse
import torch
from datasets.flower102 import FlowerDataset
import torchvision.transforms as transforms
import os
import sys
from easydict import EasyDict
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))


cfg = EasyDict() # 访问属性的方式去使用key-value 即通过key 或者 value

cfg.model_name = "resnet18"
# cfg.model_name = "vgg16_bn"
# cfg.model_name = "se_resnet50"

cfg.mixup = True  # 是否采用mixup
cfg.mixup_alpha = 1.  # beta分布的参数. beta分布是一组定义在(0,1) 区间的连续概率分布。
cfg.label_smooth = True  # 是否采用标签平滑
cfg.label_smooth_eps = 0.01  # 标签平滑超参数 eps

data_dir = "/Users/kenton/Downloads/deeplearning_dataset"
cfg.path_resnet18 = os.path.join(data_dir, "pretrained_model", "resnet18-f37072fd.pth")
cfg.path_vgg16bn = os.path.join(data_dir, "pretrained_model", "vgg16_bn-6c64b313.pth")
cfg.path_se_res50 = os.path.join(data_dir, "pretrained_model", "seresnet50-60a8950a85b2b.pkl")

# 训练参数
cfg.train_bs = 16 # batchsize
cfg.valid_bs = 16
cfg.workers = 0 #线程个数

cfg.lr_init = 0.01
cfg.momentum = 0.9
cfg.weight_decay = 1e-4 # 权重衰减

cfg.factor = 0.1 # 权重更新的比例
cfg.milestones = [30, 45] # 什么时候下降学习率
cfg.max_epoch = 50

cfg.log_interval = 10 # 日志打印间隔

# 1. 数据集
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225] # imageNet统计得到的
normTransform = transforms.Normalize(norm_mean, norm_std)

cfg.transforms_train = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    normTransform,
])

cfg.transforms_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normTransform,
])



