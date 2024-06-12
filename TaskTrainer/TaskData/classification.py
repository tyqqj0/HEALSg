# -*- CODING: UTF-8 -*-
# @time 2024/6/11 下午8:10
# @Author tyqqj
# @File classification.py
# @
# @Aim

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from TaskTrainer.basic import BasicTask


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision


#TODO: 单独放

class Cifar10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cifar10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, tgt = super(Cifar10, self).__getitem__(index)  # s
        return img, tgt, {'index': index}


class Mnist(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Mnist, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        return super(Mnist, self).__getitem__(index), {'index': index}


class ClassificationData(BasicTask):
    config_json = ('TaskData\\Classification.json')

    # def __init__(self, task='', method='', use_wandb=True, experiment_name='train', group_name='basic', device='cuda'):
    #     super(ClassificationData, self).__init__(task=task, method=method, use_wandb=use_wandb, experiment_name=experiment_name, group_name=group_name, device=device)

    def build_dataloader(self):
        # create a training data loader
        if self.args.data_set == 'cifar10':

            # 定义数据预处理
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
                transforms.RandomCrop(32, padding=4),  # 随机裁剪
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                # transforms.RandomRotation(15),  # 随机旋转
            ])

            # 载入 CIFAR-10 数据集
            train_dataset = Cifar10(root=self.args.data_root, train=True, download=True, transform=transform)
            test_dataset = Cifar10(root=self.args.data_root, train=False, download=True, transform=transform)

            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
            val_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        elif self.args.data_set == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            train_dataset = Mnist(root=self.args.data_root, train=True, transform=transform, download=True)
            test_dataset = Mnist(root=self.args.data_root, train=False, transform=transform, download=True)

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
            val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

        return train_loader, val_loader
