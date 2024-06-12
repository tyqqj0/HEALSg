# -*- CODING: UTF-8 -*-
# @time 2024/6/6 下午7:07
# @Author tyqqj
# @File main.py
# @
# @Aim 

import numpy as np
import torch
from torch.utils import data
from torch import nn

import TaskTrainer
from TaskTrainer.TaskData.classification import ClassificationData
from TaskTrainer.TaskMethod.supervised import Supervised
from TaskTrainer.TaskModel.Resnet import Resnet

from utils.arg import ConfigParser


class test_trainer(Supervised, ClassificationData, Resnet):
    # config_json = 'TaskTrainer\\Basic.json'

    def __init__(self, config_json=None, task='test', method='', use_wandb=True, experiment_name='debug', group_name='basic', device='cuda'):
        super(test_trainer, self).__init__(config_json=config_json, task=task, method=method, use_wandb=use_wandb, experiment_name=experiment_name, group_name=group_name,
                                           device=device)


if __name__ == '__main__':
    trainer = test_trainer(device='cuda') #, use_wandb=False
    trainer.run()

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision
