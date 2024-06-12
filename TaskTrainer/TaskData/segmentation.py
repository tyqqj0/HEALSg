# -*- CODING: UTF-8 -*-
# @time 2024/6/6 下午7:52
# @Author tyqqj
# @File segmentation.py
# @
# @Aim
import os
from abc import ABC, abstractmethod

from monai import data
from monai import transforms
from monai.data import load_decathlon_datalist
# from torch.utils import data
from torch.utils.data import Sampler

from TaskTrainer.basic import BasicTask
from utils.text import text_in_box


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision


class Segmentation3DData(BasicTask, ABC):
    config_json = 'TaskData\\Segmentation.json'

    # def __init__(self, config_json=None):
    #     super(Segmentation3DData, self).__init__()

    def build_dataloader(self):
        # create a training data loader
        data_dir = self.args.data_dir
        datalist_json = os.path.join(data_dir, self.args.json_list)
        train_transform = transforms.Compose(  # 一系列的数据增强操作，compose是将多个操作组合起来
            [
                transforms.LoadImaged(keys=["image", "label"]),  # 读取图像和标签
                transforms.AddChanneld(keys=["image", "label"]),  # 增加通道维度
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),  # 调整方向，RAS是右手坐标系
                transforms.Spacingd(  # 调整像素间距
                    keys=["image", "label"], pixdim=(self.args.space_x, self.args.space_y, self.args.space_z), mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityRanged(  # 调整像素值范围，将像素值范围调整到[0,1]
                    keys=["image"], a_min=self.args.a_min, a_max=self.args.a_max, b_min=self.args.b_min, b_max=self.args.b_max, clip=True
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),  # 剪裁图像
                transforms.RandCropByPosNegLabeld(  # 随机裁剪, 大小为roi_x, roi_y, roi_z，全是96， 另外，正样本和负样本的比例为1:1，样本数量为4
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(self.args.roi_x, self.args.roi_y, self.args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=self.args.RandFlipd_prob, spatial_axis=0),  # 随机翻转
                transforms.RandFlipd(keys=["image", "label"], prob=self.args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=self.args.RandFlipd_prob, spatial_axis=2),
                transforms.RandRotate90d(keys=["image", "label"], prob=self.args.RandRotate90d_prob, max_k=3),  # 随机旋转90度
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=self.args.RandScaleIntensityd_prob),  # 随机缩放
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=self.args.RandShiftIntensityd_prob),  # 随机平移
                transforms.ToTensord(keys=["image", "label"]),  # 转换为tensor，因为之前的操作都是对numpy数组进行的
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=(self.args.space_x, self.args.space_y, self.args.space_z), mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=self.args.a_min, a_max=self.args.a_max, b_min=self.args.b_min, b_max=self.args.b_max, clip=True
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )

        if self.args.test_mode:
            # 测试
            test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)  # 加载测试数据集
            test_ds = data.Dataset(data=test_files, transform=val_transform)
            test_sampler = Sampler(test_ds, shuffle=False) if self.args.distributed else None
            test_loader = data.DataLoader(
                test_ds,
                batch_size=1,
                shuffle=False,
                num_workers=self.args.workers,
                sampler=test_sampler,
                pin_memory=True,
                persistent_workers=True,
            )
            loader = test_loader
        else:
            # 如果不是测试，那么就是训练
            # 此处的datalist是一个列表，列表中的每个元素是一个字典，字典中包含了图像和标签的路径
            datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
            if self.args.use_normal_dataset:
                # 如果使用普通的数据集，那么就不需要缓存
                train_ds = data.Dataset(data=datalist, transform=train_transform)
            else:
                # 如果使用缓存数据集，那么就需要缓存
                train_ds = data.CacheDataset(
                    data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=self.args.workers
                )
            # train_ds是一个数据集，包含了训练数据和标签，transform是对数据集进行的一系列操作
            train_sampler = Sampler(train_ds) if self.args.distributed else None
            train_loader = data.DataLoader(  # 创建训练数据集的加载器
                train_ds,
                batch_size=self.args.batch_size,
                shuffle=(train_sampler is None),
                num_workers=self.args.workers,
                sampler=train_sampler,
                pin_memory=True,
                persistent_workers=True,
            )
            val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
            val_ds = data.Dataset(data=val_files, transform=val_transform)
            val_sampler = Sampler(val_ds, shuffle=False) if self.args.distributed else None
            val_loader = data.DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                num_workers=self.args.workers,
                sampler=val_sampler,
                pin_memory=True,
                persistent_workers=True,
            )
            # loader = [train_loader, val_loader]
            self.train_loader = train_loader
            self.val_loader = val_loader

    # def run_epoch(self):
    #     epoch = self.epoch
    #     text_in_box(f'Epoch {epoch}', 65)
    #
    # @abstractmethod
    # def _train_epoch(self):
    #     pass
    #
    # @abstractmethod
    # def _val_epoch(self):
    #     pass
