# -*- CODING: UTF-8 -*-
# @time 2024/6/11 下午8:32
# @Author tyqqj
# @File __init__.py
# @
# @Aim 

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

# import TaskData
# import TaskMethod
# import TaskModel
# import basic
from TaskTrainer.basic import BasicTask


def get_trainer(method: BasicTask, data: BasicTask, model: BasicTask, config_dict: dict = None, config_json=None, use_wandb=True, api_key=None, experiment_name='debug',
                group_name='basic', run_name='run'):
    class trainer(method, data, model):
        # method, data, model
        # config_json = 'TaskTrainer\\Basic.json'

        def __init__(self, config_dict=config_dict, config_json=config_json, use_wandb=use_wandb, api_key=api_key, experiment_name=experiment_name,
                     group_name=group_name, run_name=run_name):
            super(trainer, self).__init__(config_dict, config_json=config_json, use_wandb=use_wandb, api_key=api_key, experiment_name=experiment_name,
                                          group_name=group_name, custom_run_name=run_name)
