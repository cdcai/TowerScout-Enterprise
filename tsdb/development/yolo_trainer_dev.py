# Databricks notebook source
# MAGIC %pip install ultralytics

# COMMAND ----------

from ultralytics import YOLO
yolo_model = YOLO("yolov5nu.pt")

# COMMAND ----------

import torch
from torch import nn, optim
from torch import Tensor
from torchvision import transforms, datasets
from efficientnet_pytorch import EfficientNet
from enum import Enum, auto
from collections import namedtuple
from tsdb.ml.model_trainer import Metrics

class YoloModelTrainer:
    def __init__(self, optimizer_args, metrics=None, criterion: str = "MSE", model: str= "yolov5nu.pt"):
        self.model = YOLO(model)

        if metrics is None:
            metrics = [Metrics.MSE]
        self.metrics = metrics

        optimizer = self.get_optimizer()
        self.optimizer = optimizer(self.model.parameters(), **optimizer_args)

        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)
        self.loss = 0
        self.val_loss = 0
        self.threshold = 0.5
