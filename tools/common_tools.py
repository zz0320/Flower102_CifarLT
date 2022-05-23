# -*- coding: utf-8 -*-
"""
# @file name    :common_tools.py
# @author       :zz0320
# @data         :2022-4-2
# @brief        :常用文件 包含随机种子固定/检查数据目录/Logger类/绘制混淆矩阵/绘制loss和acc曲线
"""
import logging
import os
import torchvision.transforms as transforms
import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import random
import pickle
import psutil
from PIL import Image
from datetime import datetime
from models.vgg_tv import vgg16_bn
from models.se_resnet import se_resnet50
from torchvision.models import resnet18
import torch.nn as nn

def setup_seed(seed = 12345):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # cuda也有一个random
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = True

def check_data_dir(path_tmp):
    assert os.path.exists(path_tmp), \
        "\n\n路径不存在，当前变量中指定的路径是：\n{}\n请检查相对路径的设置，或者文件是否存在".format(os.path.abspath)


class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        # 初始化
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Header 输出到硬盘当中
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 配置文件Header 输出到屏幕流上
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
def make_logger(out_dir):
    """
    在out_dir文件夹下以当前时间命名，创建日志文件夹，并创建logger用于记录信息
    :param out_dir:
    :return:
    """
    # 创建log文件夹
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(out_dir, "..", "..", "results", time_str) # 为了相对路径一定找的准确
        # 根据config中的创建时间作为文件夹名字
    if not os.path.exists(log_dir):
        os.makedirs((log_dir))

    # 创建logger
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()
    return logger,log_dir

def get_model(cfg, cls_num, logger):
    """
    创建模型
    :param cfg:
    :param cls_num:
    :return:
    """
    if cfg.model_name == "resnet18":
        model = resnet18()
        if os.path.exists(cfg.path_resnet18):
            pretrained_state_dict = torch.load(cfg.path_resnet18, map_location="cpu")
            model.load_state_dict(pretrained_state_dict)    # load pretrain model
            logger.info("load pretrained model!")
        # 修改最后一层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, cls_num)  # 102
    elif cfg.model_name == "vgg16_bn":
        model = vgg16_bn()
        if os.path.exists(cfg.path_vgg16bn):
            pretrained_state_dict = torch.load(cfg.path_vgg16bn, map_location="cpu")
            model.load_state_dict(pretrained_state_dict)    # load pretrain model
            logger.info("load pretrained model!")
        # 替换网络层
        in_feat_num = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_feat_num, cls_num)
    elif cfg.model_name == "se_resnet50":
        model = se_resnet50()
        if os.path.exists(cfg.path_se_res50):
            model.load_state_dict(torch.load(cfg.path_se_res50))    # load pretrain model
            logger.info("load pretrained model!")
        in_feat_num = model.fc.in_features
        model.fc = nn.Linear(in_feat_num, cls_num)
    else:
        raise Exception("Invalid model name. got {}".format(cfg.model_name))
    return model

def show_confMat(confusion_mat, classes, set_name, out_dir, epoch=999, verbose=False, figsize=None, perc=False):
    """
    混淆矩阵绘制并保存图片
    :param confusion_mat: np.array
    :param classes: list or tuple, 类别名称
    :param set_name: str 数据集名称 train or valid or test
    :param out_dir: str 图片要保存的文件夹
    :param epoch: int 第几个epoch
    :param verbose: bool 是否打印精度信息
    :param figsize:
    :param perc:是否采用百分比，图像分割时使用 因分类数目过大
    :return:
    """
    cls_num = len(classes)

    # 归一化
    confusion_mat_tmp = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 设置图像大小
    if cls_num < 10:
        figsize = 6
    elif cls_num >= 100:
        figsize = 30
    else:
        figsize = np.linspace(6, 30, 91)[cls_num-10]
    plt.figure(figsize=(int(figsize), int(figsize*1.3)))

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys') # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_tmp, cmap=cmap)
    plt.colorbar(fraction=0.03)

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title("Confusion_Matrix_{}_{}".format(set_name, epoch))

    # 打印数字
    if perc:
        cls_per_nums = confusion_mat.sum(axis=0)
        conf_mat_per = confusion_mat / cls_per_nums
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s="{:.0%}".format(conf_mat_per[i, j]), va='center', ha='center', color='red',
                         fontsize=10)
    else:
            for i in range(confusion_mat_tmp.shape[0]):
                for j in range(confusion_mat_tmp.shape[1]):
                    plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, "Confusion_Matrix_{}.png").format(set_name))
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i]))))


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线和acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode: 'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()

if __name__ == "__main__":
    # setup_seed(2)
    # print(np.random.randint(0, 10, 1))
    logger = Logger('./logtest.log')
    logger = logger.init_logger()
    for i in range(10):
        logger.info('test:' + str(i))

    from config.flower_config import cfg
    logger.info(cfg)



















