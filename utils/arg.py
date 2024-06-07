# -*- CODING: UTF-8 -*-
# @time 2024/6/6 下午7:49
# @Author tyqqj
# @File arg.py
# @
# @Aim
import argparse
import json

import numpy as np
import torch
from torch.utils import data
from torch import nn

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

class ConfigParser:
    def __init__(self, json_file):
        self.json_file = json_file
        self.config = self.load_json(json_file)

    def load_json(self, json_file):
        with open(json_file, 'r') as f:
            config = json.load(f)
        return config

    def parse_args(self):
        parser = argparse.ArgumentParser(description=self.json_file)

        # 遍历JSON文件中的所有键值对,并添加到参数解析器中
        for key, value in self.config.items():
            if isinstance(value, bool):
                parser.add_argument(f'--{key}', action='store_true', default=value, help=f'{key}')
            else:
                parser.add_argument(f'--{key}', type=type(value), default=value, help=f'{key}')

        # 解析命令行参数
        args = parser.parse_args()

        # 用解析出的参数更新原有的配置字典
        for key, value in vars(args).items():
            self.config[key] = value

        return args

    def get_config(self):
        return self.config