# HEALSg training frame
## 1. 介绍
本项目整合了常用的训练任务和方法，旨在提供一个统一的训练框架，方便用户快速上手。
本项目的主要结构为:
### 任务基本单位
- `TaskTrainer` 任务基本单位，包含了任务的基本信息，如任务名称、任务描述、任务数据集等。
- `TaskConfig` 任务配置，包含了任务的配置信息，如模型、优化器、学习率等。
- `TaskRunner` 任务运行器，包含了任务的运行信息，如训练、验证、测试等。
#### TaskTrainer
- `TaskTrainer` 的基本结构通过 `BasicTask` 接口定义，实际任务通过继承 `BasicTask` 接口实现。
- `TaskTrainer` 的结构基本上可以分为三个结构，分别是 `TaskData`、`TaskModel` 和 `TaskMethod`, 其中 `TaskData` 用于定义数据集，`TaskModel` 用于定义模型，`TaskMethod` 用于定义训练方法。项目实现了一些常用的 `TaskData`、`TaskModel` 和 `TaskMethod`，用户可以通过继承这些类实现自己的任务。
## 2. 训练
如果使用wandb，需要在 `main.py` 中将 `api_key` 设置为自己的 `api_key`。
```shell
python main.py
```
如果不使用wandb，需要在 `main.py` 中将 `use_wandb` 设置为 `False`。
## 1. Introduction
