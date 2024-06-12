# -*- CODING: UTF-8 -*-
# @time 2024/6/6 下午7:49
# @Author tyqqj
# @File arg.py
# @
# @Aim
import argparse
import json
import os

import numpy as np
import torch
from torch.utils import data
from torch import nn

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

import json
import argparse


class ConfigParser:
    def __init__(self, json_file: str = None, config_dict: dict = None):
        self.different = None
        if config_dict is None:
            config_dict = {}
        self.json_file = json_file
        self.config_dict = config_dict
        self.config = self.load_config()

    def load_config(self):
        config = {}
        if self.json_file:
            # 如果不存在，询问是否创建
            if not os.path.exists(self.json_file):
                print(f'{self.json_file} not exist, create it?')
                if input('y/n: ') == 'y':
                    with open(self.json_file, 'w') as f:
                        json.dump({}, f)
                else:
                    raise FileNotFoundError(f'{self.json_file} not exist')

            with open(self.json_file, 'r') as f:
                config = json.load(f)
        if self.config_dict:
            config.update(self.config_dict)
        return config

    def parse_args(self):
        parser = argparse.ArgumentParser(description=self.json_file)  # '配置参数解析器'

        for key, value in self.config.items():
            if isinstance(value, bool):
                parser.add_argument(f'--{key}', action='store_true', default=value, help=f'{key}的值')
            else:
                parser.add_argument(f'--{key}', type=type(value), default=value, help=f'{key}的值')

        args = parser.parse_args()

        self.different = {}
        for key, value in vars(args).items():
            if self.config[key] != value:
                self.different[key] = value
            self.config[key] = value

        return args

    def update_config(self, new_config):
        self.config.update(new_config)

    def get_different(self):
        return self.different

    def get_config(self):
        return self.config
