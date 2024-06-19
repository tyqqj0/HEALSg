# -*- CODING: UTF-8 -*-
# @time 2024/6/12 下午7:07
# @Author tyqqj
# @File supervised.py
# @
# @Aim 

import numpy as np
import torch
from torch.utils import data
from torch import nn

from TaskTrainer.basic import BasicEpoch


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
        for inputs, targets, addition in self.loader:
            # print(batch_idx)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs, add_infos = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.shape[0]
            # print(total)
            correct += predicted.eq(targets).sum().item()
        if self.scheduler is not None:
            self.scheduler.step()
        loss = train_loss / total
        crr = 100. * correct / total
        return {'loss': loss, 'acc': crr}


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
                total += targets.shape[0]
                correct += predicted.eq(targets).sum().item()
        return {'loss': val_loss / total, 'acc': 100. * correct / total}


class TrainEpochSeg(BasicEpoch):
    def __init__(self, loader, task):
        super(TrainEpochSeg, self).__init__('train', loader, task, 'red', bar=True)

    def epoch(self):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets, addition in self.loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs, add_infos = self.sliding_window_inference(self.model, inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()


        if self.scheduler is not None:
            self.scheduler.step()



class ValEpochSeg(BasicEpoch):
    def __init__(self, loader, task):
        from .utils.inferers import SlidingWindowInferer
        super(ValEpochSeg, self).__init__('val', loader, task, 'blue', bar=True)
        self.sliding_window_inference = SlidingWindowInferer

    def epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, addition) in enumerate(self.loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, add_infos = self.sliding_window_inference(self.model, inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.shape[0]
                correct += predicted.eq(targets).sum().item()
        return {'loss': val_loss / total, 'acc': 100. * correct / total}
