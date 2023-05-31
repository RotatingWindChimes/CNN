"""

数据集加载与处理，数据批量的获取

"""


from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def load_data():
    """ 数据加载与获取 """
    data_path = os.path.join("data", "CIFAR-100")

    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])

    # 训练集和测试集
    cifar_train = datasets.CIFAR100(root=data_path, download=True, train=True, transform=transform_train)
    cifar_test = datasets.CIFAR100(root=data_path, download=True, train=False, transform=transform_test)

    # 数据批量
    train_iter = DataLoader(cifar_train, batch_size=128, shuffle=True)
    test_iter = DataLoader(cifar_test, batch_size=128, shuffle=False)

    return train_iter, test_iter
