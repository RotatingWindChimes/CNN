"""

图像增强操作

"""

import torch
import numpy as np


# 定义 CutMix 函数
def cutmix_data(inputs, targets, alpha=1.0):
    """
    生成 CutMix 数据
    inputs: 输入图像的张量
    targets: 输入图像的标签
    alpha: CutMix 操作的参数，控制混合程度

    cutmix操作，通过将多个图像的一部分混合在一起来生成新的训练样本
    """

    # 批量大小
    batch_size = inputs.shape[0]

    # 打乱输入顺序
    indices = torch.randperm(batch_size)
    shuffled_inputs = inputs[indices]
    shuffled_targets = targets[indices]

    # 随机生成切割区域的掩码
    lam = np.random.beta(alpha, alpha)

    # 计算要切割的区域的大小
    cut_h = int(inputs.shape[2] * lam)
    cut_w = int(inputs.shape[3] * lam)

    # 计算切割区域的中心点的横纵坐标
    cx = np.random.randint(0, inputs.shape[3])
    cy = np.random.randint(0, inputs.shape[2])

    # 计算切割区域的坐标
    bbx1 = np.clip(cx - cut_w // 2, 0, inputs.size(3))
    bbx2 = np.clip(cx + cut_w // 2, 0, inputs.size(3))
    bby1 = np.clip(cy - cut_h // 2, 0, inputs.size(2))
    bby2 = np.clip(cy + cut_h // 2, 0, inputs.size(2))

    # 将切割区域填充为另一个样本的像素值
    new_inputs = inputs.clone()
    new_inputs[:, :, bby1:bby2, bbx1:bbx2] = shuffled_inputs[:, :, bby1:bby2, bbx1:bbx2]

    # 计算新的标签
    new_targets = (1 - (bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(2) * inputs.size(3))) * targets + \
                  ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(2) * inputs.size(3))) * shuffled_targets

    return new_inputs, new_targets.long()


def cutout_data(images, length):
    """
    生成 Cutout 图像
    images: 输入图像的张量
    length: 需要遮挡的边长

    cutout操作，通过随机遮挡图像的一部分来增加训练样本的多样性。
    """

    # 图像维度信息
    batch_size, channels, height, width = images.size()

    # 掩码范围
    mask = torch.ones((batch_size, channels, height, width), dtype=torch.float32).to(device=images.device)
    y = torch.randint(height, size=(batch_size,))
    x = torch.randint(width, size=(batch_size,))

    # 遮挡区域的边界
    y1 = torch.clamp(y - length // 2, 0, height)
    y2 = torch.clamp(y + length // 2, 0, height)
    x1 = torch.clamp(x - length // 2, 0, width)
    x2 = torch.clamp(x + length // 2, 0, width)

    for i in range(batch_size):
        mask[i, :, y1[i]:y2[i], x1[i]:x2[i]] = 0.0

    masked_images = images * mask

    return masked_images


def mixup_data(x, y, alpha=0.8):
    """
    生成 Mixup 图像

    x: 输入图像张量
    y: 输入图像对应的标签
    alpha: 混合程度
    Mixup操作，通过将不同样本的输入和标签进行线性混合，生成新的样本对，以扩充训练数据集
    """

    batch_size = x.size(0)
    weights = np.random.beta(alpha, alpha, size=batch_size)
    weights = torch.from_numpy(weights).float().to(x.device)

    index = torch.randperm(batch_size).to(x.device)

    mixed_x = weights.view(-1, 1, 1, 1) * x + (1 - weights).view(-1, 1, 1, 1) * x[index]
    mixed_y = weights * y + (1 - weights) * y[index]

    return mixed_x, mixed_y.long()


import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

data_path = os.path.join("data", "CIFAR-100")

cifar_train = datasets.CIFAR100(root=data_path, download=True, train=True, transform=transforms.ToTensor())
cifar_test = datasets.CIFAR100(root=data_path, download=True, train=False, transform=transforms.ToTensor())

train_iter = DataLoader(cifar_train, batch_size=6, shuffle=True)
test_iter = DataLoader(cifar_test, batch_size=6, shuffle=False)

X, y = next(iter(train_iter))

_, axes = plt.subplots(nrows=2, ncols=6)
plt.figure(figsize=(1.5, 0.5))

for i in range(6):
    data = X[i].permute(1, 2, 0).numpy()
    ax = axes[0][i]
    ax.imshow(data, interpolation='nearest')

    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_title("Origin_figure")

for i in range(6):
    X, y = mixup_data(X, y)
    data = X[i].permute(1, 2, 0).numpy()
    ax = axes[1][i]
    ax.imshow(data, interpolation='nearest')

    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_title("Mixup_figure")

plt.show()
