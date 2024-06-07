# -*- CODING: UTF-8 -*-
# @time 2024/6/6 下午7:52
# @Author tyqqj
# @File segmentation.py
# @
# @Aim
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils import data
from torch import nn

from utils.arg import ConfigParser
from .basic import BasicTask

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision


class SegmentationTask(BasicTask, ABC):
    def __init__(self, config_json='./SegmentationDefault.json'):
        super(SegmentationTask, self).__init__()
        self.config = ConfigParser(config_json, self.config).get_config()

    @abstractmethod
    def build_model(self):
        pass
