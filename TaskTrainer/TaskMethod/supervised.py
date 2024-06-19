# -*- CODING: UTF-8 -*-
# @time 2024/6/11 下午8:35
# @Author tyqqj
# @File supervised.py
# @
# @Aim 

import numpy as np
import torch
from torch.utils import data
from torch import nn

from TaskTrainer.TaskMethod.epochs.supervised import TrainEpoch, ValEpoch, ValEpochSeg
from TaskTrainer.basic import BasicTask, BasicEpoch


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

# TODO: epoch单独放


class Supervised(BasicTask):
    config_json = 'TaskMethod\\Supervised.json'

    def __init__(self, config_dict, config_json=None, custom_run_name='', use_wandb=True, api_key=None, experiment_name='train', group_name='basic'):
        super(Supervised, self).__init__(config_dict, config_json=config_json, use_wandb=use_wandb, api_key=api_key, experiment_name=experiment_name,
                                         custom_run_name=custom_run_name, group_name=group_name)
        self.train_epoch = TrainEpoch(self.train_loader, self)
        self.val_epoch = ValEpoch(self.val_loader, self)

    # def build_optimizer(self):
    #     # 定义优化器
    #     optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
    #     return optimizer

    def build_optimizer(self):
        # 定义优化器
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        return optimizer

    def build_scheduler(self):
        # 定义学习率调整策略
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        return scheduler

    def run_epoch(self):
        train_log = self.train_epoch.run()
        val_log = self.val_epoch.run()
        return {'train': train_log, 'val': val_log}


class SupervisedSegmentation(BasicTask):
    config_json = 'TaskMethod\\SupervisedWithInferer.json'

    def __init__(self, config_dict, config_json=None, custom_run_name='', use_wandb=True, api_key=None, experiment_name='train', group_name='basic'):
        super(SupervisedSegmentation, self).__init__(config_dict, config_json=config_json, use_wandb=use_wandb, api_key=api_key, experiment_name=experiment_name,
                                                     custom_run_name=custom_run_name, group_name=group_name)
        self.train_epoch = TrainEpoch(self.train_loader, self)
        self.val_epoch = ValEpochSeg(self.val_loader, self)

    def build_optimizer(self):
        # 定义优化器
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        return optimizer

    def build_scheduler(self):
        # 定义学习率调整策略
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        return scheduler

    def run_epoch(self):
        train_log = self.train_epoch.run()
        val_log = self.val_epoch.run()
        return {'train': train_log, 'val': val_log}
