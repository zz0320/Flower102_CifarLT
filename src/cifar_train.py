# -*- coding: utf-8 -*-
"""
# @file name    :cifar_train.py
# @author       :zz0320
# @data         :2022-4-12
# @brief        :cifar-10训练代码
"""

import os
import sys
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tools.model_trainer import ModelTrainer
from tools.mixup import mixup_data, mixup_criterion
from tools.common_tools import *
from tools.my_loss import LabelSmoothLoss
from models.resnet_cifar10 import resnet20
from config.cifar_config import cfg
from datetime import datetime
from datasets.cifar_longtail import CifarLTDataset
from tools.progressively_balance import ProgressiveSampler


setup_seed(12345) # 先固定随机种子
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# 解析命令行的参数
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr',default=None, type=float, help='learning rate')
parser.add_argument('--bs',default=None, type=int, help='training batch size')
parser.add_argument('--max_epoch',default=None, type=int)
parser.add_argument('--data_root_dir',default=r"/Users/kenton/Downloads/deeplearning_dataset/cifar-10",
                    type=str, help='path to your dataset')

args = parser.parse_args()

# 接收参数
cfg.lr_init = args.lr if args.lr else cfg.lr_init
cfg.train_bs = args.bs if args.bs else cfg.train_bs
cfg.max_epoch = args.max_epoch if args.bs else cfg.max_epoch


if __name__ == '__main__':
    # step 0: config
    # 数据路径
    train_dir = os.path.join(args.data_root_dir, "cifar10_train")
    valid_dir = os.path.join(args.data_root_dir, "cifar10_test")
    # path_state_dict = "/Users/kenton/Downloads/deeplearning_dataset/pretrain_model/resnet18-f37072fd.pth" # 预训练模型所在位置
    check_data_dir(train_dir) # 验证路径是否存在
    check_data_dir(valid_dir)

    # 创建logger
    res_dir = os.path.join(BASE_DIR, "..", "..", "results")
    logger, log_dir = make_logger(res_dir)

    # step 1: 数据集
    # 构建Dataset实例，构建DataLoader
    train_data = CifarLTDataset(root_dir=train_dir, transform=cfg.transforms_train,isTrain=True)
    valid_data = CifarLTDataset(root_dir=valid_dir, transform=cfg.transforms_valid,isTrain=False)
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers)
    valid_loader = DataLoader(dataset=valid_data, batch_size=cfg.valid_bs, num_workers=cfg.workers)
    if cfg.pb:
        sampler_generator = ProgressiveSampler(train_data, cfg.max_epoch)

    # 2. 模型
    model = resnet20()
    """
    # 加载预训练模型的参数 state_dict
    if os.path.exists(path_state_dict):
        pretrained_state_dict = torch.load(path_state_dict, map_location='cpu')
        model.load_state_dict(pretrained_state_dict)
        logger.info("Load pretrained model")
    else:
        logger.info("The pretrained model path {} is not exists".format(path_state_dict))
    
    # 修改最后一层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, train_data.cls_num)
    # to device
    """
    model.to(device)

    # 3. 损失函数 优化器 等
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr_init, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    # 4. 迭代训练
    loss_rec = {'train':[], "valid":[]}
    acc_rec = {'train':[], "valid":[]}
    best_acc, best_epoch = 0, 0
    for epoch in range(cfg.max_epoch):
        if cfg.pb:
            sampler, _ = sampler_generator(epoch)
            train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=False,
                                      num_workers=cfg.workers, sampler=sampler)
        # dataloader
        loss_train, acc_train, mat_train, path_error_train = ModelTrainer.train(
            train_loader, model, loss_f, optimizer, scheduler, epoch, device, cfg, logger)

        # valid
        loss_valid, acc_valid, mat_valid, path_error_valid = ModelTrainer.valid(
            valid_loader, model, loss_f, device)

        logger.info("Epoch[{:0>3}/{:0>3}] Train Acc:{:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(epoch + 1,
                cfg.max_epoch, acc_train, acc_valid, loss_train, loss_valid,
                optimizer.param_groups[0]["lr"]))

        scheduler.step() # 学习率进行更新！！！

        # 记录训练信息
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        # 保存混淆矩阵图
        show_confMat(mat_train, train_data.names, "train", log_dir, epoch=epoch, verbose=epoch == cfg.max_epoch - 1)
        show_confMat(mat_valid, valid_data.names, "valid", log_dir, epoch=epoch, verbose=epoch == cfg.max_epoch - 1)
        # 保存loss曲线 acc曲线
        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        # 保存模型
        if best_acc < acc_valid or epoch == cfg.max_epoch - 1:
            best_epoch = epoch if best_acc < acc_valid else best_epoch
            best_acc = acc_valid if best_acc < acc_valid else best_acc
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}
            pkl_name = "checkpoint_{}.pkl".format(epoch) if epoch == cfg.max_epoch - 1 else "checkpoint_best.pkl"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)

            # 保存错误图片的路径
            err_ims_name = "error_imgs_{}.pkl".format(epoch) if epoch == cfg.max_epoch - 1 else "error_imgs_best.pkl"
            path_err_img = os.path.join(log_dir, err_ims_name)
            error_info = {}
            error_info["train"] = path_error_train
            error_info["valid"] = path_error_valid
            pickle.dump(error_info, open(path_err_img, "wb"))
    logger.info("{} done, best acc:{} in: {}".format(
        datetime.strftime(datetime.now(), "%m-%d_%H-%M"), best_acc, best_epoch))