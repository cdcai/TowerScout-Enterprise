# Databricks notebook source
# MAGIC %pip install ultralytics
# MAGIC %pip install efficientnet_pytorch

# COMMAND ----------



# COMMAND ----------

from ultralytics import YOLO

yolo_model = YOLO("yolov5nu.pt")

# COMMAND ----------

import torch
from torch import nn, optim
from torch import Tensor
from torchvision import transforms, datasets
from enum import Enum, auto
from collections import namedtuple
from tsdb.ml.model_trainer import Metrics
from ultralytics.utils.torch_utils import TORCH_2_4

"""
Note that I have removed torch.nn.parallel.DistributedDataParallel (DDP) usage that was present in 
Ultralytics since we use Hyperopt for distributed tuning and it's not clear how they will interact with each other
"""

class YoloModelTrainer:
    def __init__(self, optimizer_args, metrics=None, criterion: str = "MSE", model: str= "yolov5nu.pt"):
        self.model = YOLO(model)

        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )

        if metrics is None:
            metrics = [Metrics.MSE]
        self.metrics = metrics

        optimizer = self.get_optimizer()
        self.optimizer = optimizer(self.model.parameters(), **optimizer_args)

        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)
        

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        # remove self.ema conditional

    def training_step(self, minibatch, **kwargs) -> dict:
        self.model.train()
        
        # forward pass
        # preprocess batch is called in source code but just returns batch
        self.loss, self.loss_items = self.model(minibatch)
        # self.tloss = (
        #                 (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
        #             )

        # backward pass
        self.scaler.scale(self.loss).backward()

        # Note that in ultralytics, losses are accumulated between N batchs before calling optimizer_step
        self.optimizer_step()

        #######
        logits, images, labels = forward_func(self.model, minibatch)
        loss = self.criterion(logits, images)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return score(logits, labels, Steps["TRAIN"].name, self.metrics)
