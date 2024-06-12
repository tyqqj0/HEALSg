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

from TaskTrainer.basic import BasicTask, BasicEpoch


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

class TrainEpoch(BasicEpoch):
    def __init__(self, loader, task):
        super(TrainEpoch, self).__init__('train', loader, task, 'red', bar=True)

    def epoch(self):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, addition) in enumerate(self.loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs, add_infos = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        if self.scheduler is not None:
            self.scheduler.step()
        return {'loss': train_loss / total, 'acc': 100. * correct / total}


class ValEpoch(BasicEpoch):
    def __init__(self, loader, task):
        super(ValEpoch, self).__init__('val', loader, task, 'blue', bar=True)

    def epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, addition) in enumerate(self.loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, add_infos = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return {'loss': val_loss / total, 'acc': 100. * correct / total}


class Supervised(BasicTask):
    config_json = 'TaskMethod\\Supervised.json'

    def __init__(self, config_json=None, task='', method='', use_wandb=True, experiment_name='train', group_name='basic', device='cuda'):
        super(Supervised, self).__init__(config_json=config_json, task=task, method=method, use_wandb=use_wandb, experiment_name=experiment_name, group_name=group_name, device=device)
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
