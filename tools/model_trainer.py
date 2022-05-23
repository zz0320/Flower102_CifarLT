# -*- coding: utf-8 -*-
"""
# @file name    :model_trainer.py
# @author       :zz0320
# @data         :2022-3-31
# @brief        :工具函数 用来模型训练 然后输出训练中的数据
"""

import os
import sys
import torch
import numpy as np
from collections import Counter
from tools.mixup import mixup_data, mixup_criterion

class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, scheduler, epoch_idx, device, cfg, logger):
        model.train()

        class_num = data_loader.dataset.cls_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        loss_mean = 0
        acc_avg = 0
        path_error = []
        label_list = []
        for i, data in enumerate(data_loader):

            # _, label = data
            inputs, labels, path_imgs = data
            label_list.extend(labels.tolist())

            # inputs, labels = data
            inputs, labels, path_imgs = data
            inputs, labels = inputs.to(device), labels.to(device)

            # mixup
            if cfg.mixup:
                mixup_inputs, label_a, label_b, lam = mixup_data(inputs, labels,
                                        cfg.mixup_alpha,device)
                inputs = mixup_inputs


            # forward & backward
            outputs = model(inputs)
            optimizer.zero_grad()

            # loss计算
            if cfg.mixup:
                loss = mixup_criterion(loss_f, outputs.cpu(), label_a.cpu(), label_b.cpu(), lam)

            else:
                loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward()
            optimizer.step()

            # 统计loss
            loss_sigma.append(loss.item())
            loss_mean = np.mean(loss_sigma)

            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1
                if cate_i != pre_i:
                    path_error.append((cate_i, pre_i, path_imgs[j]))
            acc_avg = conf_mat.trace() / conf_mat.sum()

            # 每 log_interval 个 iter 打印一次训练信息
            if i % cfg.log_interval == cfg.log_interval - 1:
                logger.info("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%}".format(epoch_idx\
                    + 1, cfg.max_epoch, i + 1, len(data_loader), loss_mean, acc_avg))
        logger.info("epoch:{} sampler:{}".format(epoch_idx, Counter(label_list)))
        return loss_mean, acc_avg, conf_mat, path_error

    @staticmethod
    def valid(data_loader, model, loss_f, device):
        model.eval()

        class_num = data_loader.dataset.cls_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        path_error = []

        for i, data in enumerate(data_loader):
            inputs, labels, path_imgs = data
            # inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_f(outputs.cpu(), labels.cpu())

            # 统计混淆矩阵
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1
                if cate_i != pre_i:
                    path_error.append((cate_i, pre_i, path_imgs[j]))

            # 统计loss
            loss_sigma.append(loss.item())

        acc_avg = conf_mat.trace() / conf_mat.sum()

        return np.mean(loss_sigma), acc_avg, conf_mat, path_error











