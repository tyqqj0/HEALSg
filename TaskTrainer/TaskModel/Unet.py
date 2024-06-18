# -*- CODING: UTF-8 -*-
# @time 2024/6/18 下午11:08
# @Author tyqqj
# @File Unet.py
# @
# @Aim 

import numpy as np
import torch
from torch.utils import data
from torch import nn
import monai

from TaskTrainer import BasicTask


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision


class Unet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )

    def forward(self, x):
        return self.model(x), {}


class UnetTask(BasicTask):
    config_json = 'TaskModel\\Unet.json'

    # def __init__(self, config_json=None):
    #     super(UnetTask, self).__init__()

    def build_model(self):
        # 定义网络结构
        model = Unet()
        return model

    def build_criterion(self):
        # 定义损失函数
        criterion = monai.losses.DiceLoss(sigmoid=True)
        return criterion
