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
## 3. 添加功能
要向 HEALSg 训练框架中添加新功能，首先需要理解 BasicTask 类的核心组成部分。BasicTask 作为一个抽象基类，定义了任务的三大关键组成部分：

### 3.1 理解 BasicTask 类的三大组成部分
BasicTask 类作为任务执行的核心，分为三个主要部分：

* TaskData：管理和配置数据集的部分，负责数据的加载和预处理。
* TaskModel：定义模型的结构和参数，以及损失函数。
* TaskMethod：包括训练方法和验证方法,训练流程等。

在 BasicTask 类中，有几个关键的抽象方法需要在子类中实现，以确保任务能够正常运行。

#### 1. build_dataloader()
**职责**: 此方法负责构建和返回训练和验证的数据加载器。

**实现需求**: 加载和预处理数据。

分割数据为训练集和验证集。
返回 DataLoader 对象, 注意要返回`train_loader`和`val_loader`。

**示例**:

```python
def build_dataloader(self):
    # 实现数据加载逻辑
    train_dataset = CustomDataset(self.config, train=True)
    val_dataset = CustomDataset(self.config, train=False)
    train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
    return train_loader, val_loader
```
    
#### 2. build_model()
**职责**: 构建并返回模型实例。

**实现需求**:定义模型架构。

初始化模型参数。

**示例**:
```python
def build_model(self):
    model = CustomModel()
    return model
```
#### 3. build_criterion()
**职责**: 定义并返回用于训练的损失函数。

**实现需求**:选择合适的损失函数。

**示例**:

```python
def build_criterion(self):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion
```
#### 4. build_optimizer()
**职责**: 创建并返回优化器。

**实现需求**:配置优化器参数，绑定优化器到模型参数。

**示例**:

```python
def build_optimizer(self):
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    return optimizer
```
#### 5. build_scheduler()
**职责**: 创建并返回学习率调度器。

**实现需求**:设定学习率变化策略。

**示例**:

```python
def build_scheduler(self):
    scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    return scheduler
```
#### 6. run_epoch()
**职责**: 实现单个训练周期的具体逻辑。

**实现需求**:执行一个训练周期，计算并返回相关的性能指标。

**示例**:

```python
def run_epoch(self):
    train_log = self.train_epoch.run()
    val_log = self.val_epoch.run()
    return {'train': train_log, 'val': val_log}
```

通过实现这些抽象方法，您可以根据项目需求定制化 BasicTask 的行为，从而创建多样化的机器学习任务。
## 1. Introduction
