# -*- CODING: UTF-8 -*-
# @time 2024/6/6 下午7:52
# @Author tyqqj
# @File basic.py
# @
# @Aim 

import numpy as np
import torch
from torch.utils import data
from torch import nn

from abc import ABC, abstractmethod

import wandb

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

class BasicTask(ABC):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.criterion = self.build_criterion()
        self.scheduler = self.build_scheduler()
        self.train_loader, self.val_loader = self.build_dataloader()

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def build_optimizer(self):
        pass

    @abstractmethod
    def build_criterion(self):
        pass

    @abstractmethod
    def build_scheduler(self):
        pass

    @abstractmethod
    def build_dataloader(self):
        pass
