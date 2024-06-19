# -*- CODING: UTF-8 -*-
# @time 2024/6/19 下午3:50
# @Author tyqqj
# @File inferers.py
# @
# @Aim 

import numpy as np
import torch
from torch.utils import data
from torch import nn

from monai.inferers import sliding_window_inference
# from monai.metrics


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

class SlidingWindowInferer(nn.Module):
    def __init__(self, roi_size=(128, 128), sw_batch_size=2, overlap=0.5, mode='gaussian', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode

    def forward(self, predictor, x):
        return sliding_window_inference(x, predictor=predictor, roi_size=self.roi_size, sw_batch_size=self.sw_batch_size, overlap=self.overlap, mode=self.mode)
