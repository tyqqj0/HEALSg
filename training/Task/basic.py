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
from utils.arg import ConfigParser

import wandb


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

class BasicTask(ABC):
    def __init__(self, config_json='./BasicDefault.json'):
        self.config = ConfigParser(config_json).get_config()
        self.args = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def parse_args(self):
        self.args = ConfigParser(config_dict=self.config).parse_args()

    def run(self):
        if self.args is None:
            raise ValueError('Please parse args first')



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
