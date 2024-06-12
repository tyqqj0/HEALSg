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
from TaskTrainer.TaskData.segmentation import Segmentation3DData
from TaskTrainer.TaskMethod.supervised import Supervised
from TaskTrainer.TaskModel.Resnet import Resnet

from utils.arg import ConfigParser


class test_trainer(Supervised, ClassificationData, Resnet):
    # method, data, model
    # config_json = 'TaskTrainer\\Basic.json'

    def __init__(self, config_dict=None, config_json=None, method='', use_wandb=True, experiment_name='debug', group_name='basic'):
        super(test_trainer, self).__init__(config_dict, config_json=config_json, use_wandb=use_wandb, experiment_name=experiment_name, method=method,
                                           group_name=group_name)


if __name__ == '__main__':
    trainer = test_trainer()  # , use_wandb=False
    trainer.run()

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision
