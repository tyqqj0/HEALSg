# -*- CODING: UTF-8 -*-
# @time 2024/6/11 下午8:24
# @Author tyqqj
# @File Resnet.py
# @
# @Aim 

import torch
import torchvision
from torch import nn
from torchvision.models import ResNet18_Weights

from TaskTrainer.basic import BasicTask


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

class resnet18(torch.nn.Module):
    def __init__(self, pretrained=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ResNet18_Weights.DEFAULT  # weights =
        self.model = torchvision.models.resnet18()  # (weights=weights)  # pretrained=pretrained

    def forward(self, x):
        return self.model(x), {}


class Resnet(BasicTask):
    config_json = 'TaskModel\\Resnet.json'

    # def __init__(self, config_json=None):
    #     super(Resnet, self).__init__()

    def build_model(self):
        # 定义网络结构
        model = resnet18(pretrained=False)
        return model

    def build_criterion(self):
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        return criterion
