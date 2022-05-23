# -*- coding: utf-8 -*-
"""
# @file name    :mixup.py
# @author       :zz0320
# @data         :2022-4-10
# @brief        :mixup功能实现 内含测试样例
"""

import torch
import numpy as np

def mixup_data(x, y, alpha=1.0, device=True):
    """Return mixed inputs, pairs of targets, and lambda"""

    # 通过beta分布获得lambda beta分布的参数alpha == beta 因此都是alpha
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    # 获取需要重叠的图片的标号
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    # mixup
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    path_1 = r"/Users/kenton/Downloads/deeplearning_dataset/flower102/jpg/image_00001.jpg"
    path_2 = r"/Users/kenton/Downloads/deeplearning_dataset/flower102/jpg/image_00002.jpg"

    img1 = cv2.imread(path_1)
    img2 = cv2.imread(path_2)
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))

    alpha = 1
    figsize = 15
    plt.figure(figsize=(int(figsize), int(figsize)))
    for i in range(1, 10):
        # lam = i * 0.1
        lam = np.random.beta(alpha, alpha)
        im_mixup = (img1 * lam + img2 * (1 - lam)).astype(np.uint8)
        im_mixup = cv2.cvtColor(im_mixup, cv2.COLOR_BGR2RGB)
        plt.subplot(3, 3, i)
        plt.title("lamda_{:.2f}".format(lam))
        plt.imshow(im_mixup)
    plt.show()



















