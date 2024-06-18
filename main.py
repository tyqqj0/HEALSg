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
from TaskTrainer.TaskModel.Unet import UnetTask

from utils.arg import ConfigParser


def main(experiment_name='train', group_name='basic', run_name='run', config_dict='agent', api_key=None, use_wandb=True):
    Trainer = TaskTrainer.get_trainer(Supervised, ClassificationData, Resnet, config_dict=config_dict, experiment_name=experiment_name, group_name=group_name,
                                      run_name=run_name, api_key=api_key, use_wandb=use_wandb)
    trainer = Trainer()
    trainer.run()

def segmentation(experiment_name='train', group_name='basic', run_name='run', config_dict='agent', api_key=None, use_wandb=True):
    Trainer = TaskTrainer.get_trainer(Supervised, Segmentation3DData, UnetTask, config_dict=config_dict, experiment_name=experiment_name, group_name=group_name,
                                      run_name=run_name, api_key=api_key, use_wandb=use_wandb)
    trainer = Trainer()
    trainer.run()


if __name__ == '__main__':
    experiment_name = 'train'
    group_name = 'basic'
    run_name = 't1'
    api_key = 'YOUR_API_KEY'
    use_wandb = True
    main(experiment_name=experiment_name, group_name=group_name, run_name=run_name, api_key=api_key, use_wandb=use_wandb)
