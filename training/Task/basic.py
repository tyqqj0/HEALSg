# -*- CODING: UTF-8 -*-
# @time 2024/6/6 下午7:52
# @Author tyqqj
# @File basic.py
# @
# @Aim
import time

import numpy as np
import torch
from torch.utils import data
from torch import nn

from abc import ABC, abstractmethod
from utils.arg import ConfigParser
from utils.text import text_in_box

import wandb


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision


class BasicEpoch(ABC):
    def __init__(self, task, color):
        self.task = task
        self.color = color


    def run(self):
        text_in_box(f'Epoch {self.task.epoch}', 'Epoch', True)


    @abstractmethod
    def _run_step(self):
        pass

class BasicTask(ABC):
    def __init__(self, config_json='./BasicDefault.json', task='', method='', use_wandb=True, experiment_name='train', group_name='basic'):
        #################DATA_params#################
        self.epoch = None
        self.running = None
        self.max_epoch = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        #################init_params#################

        self.different_args = None
        self.config = ConfigParser(config_json).get_config()
        self.args = None

        self.use_wandb = use_wandb
        # self.run_name = ''
        self.experiment_name = experiment_name
        self.group_name = group_name
        self.task = task
        self.method = method
        self.time = time.strftime("%Y%m%d-%H%M%S", time.localtime())

        #################build_params#################
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def parse_args(self):
        parser = ConfigParser(config_dict=self.config)
        self.args = parser.parse_args()
        self.different_args = parser.different
        # if self.use_wandb:
        #     wandb.config = self.config

    def log(self, data):
        if self.use_wandb:
            wandb.log(data, step=self.epoch, commit=True)

    # def

    # def _generate_run_name(self):
    #

    def run(self):
        if self.args is None:
            raise ValueError('Please use parse_args first')
        if self.use_wandb:
            wandb.init(project=self.experiment_name, name=self.run_name, config=self.args, group=self.group_name)
        self.running = True
        # self._run()
        for epoch in range(1, self.max_epoch + 1):
            self.epoch = epoch
            self._run_epoch()
        self.running = False
        if self.use_wandb:
            wandb.finish()

    @abstractmethod
    def _run_epoch(self):
        pass

    @property
    def run_name(self):
        optional_info = ''

        # self.run_name = self.experiment_name + '_' + self.group_name + '_'
        if self.different_args is not None:
            for key, value in self.different_args.items():
                self.run_name += f'{key}-{value}_'
        # self.run_name = self.run_name[:-1]
        # 如果运行已经开始，返回当前运行的 run_name
        if self.running:
            return
        # 每次调用 run_name 时，都会根据当前属性构造 run_name
        components = [
            self.method,
            self.args.model,
            self.args.dataset,
            optional_info,
            self.time
        ]
        # 过滤掉空字符串，确保组件不为空
        valid_components = [component for component in components if component]
        # 使用下划线连接所有非空组件
        return '_'.join(valid_components)

    @run_name.setter
    def run_name(self, value):
        # 如果需要，可以设置一个方法来允许外部修改 run_name 的一些组件
        # 这里的逻辑取决于您希望如何处理 run_name 的赋值
        pass

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
