# -*- CODING: UTF-8 -*-
# @time 2024/6/12 下午7:55
# @Author tyqqj
# @File sweep.py
# @
# @Aim 

import numpy as np
import torch
from torch.utils import data
from torch import nn

from main import main

import wandb

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

# 步骤
# 1. 定义sweep配置或在wandb网站上定义
# 2. 生成sweep_id
# 3. 运行sweep的agent
from utils.text import text_in_box

experiment_name = 'debug'
group_name = 'basic'
sweep_id = 'tyqqj_ai/debug/tc4kuxd4'

text_in_box(f'START AGENT {sweep_id}')

wandb.agent(sweep_id, function=main(experiment_name=experiment_name, group_name=group_name, config_dict='agent'), count=50)
