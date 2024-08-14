# Databricks notebook source
import torch
from torch import nn
from enum import Enum

from collections import namedtuple
from functools import partial

# COMMAND ----------

class DemoMetrics(Enum):
    MSE = nn.MSELoss()

# COMMAND ----------

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Sigmoid()
        )
    
    def forward(self, images):
        images = self.encoder(images)
        return self.decoder(images)

# COMMAND ----------

ModelOutput = namedtuple("ModelOutput", ["loss", "logits", "images"])


class ModelTrainer:
    def __init__(self, optimizer_args, metrics=None, criterion: str="MSE"):
        self.model = Autoencoder()
        
        if metrics is None:
            metrics = [DemoMetrics.MSE]
        
        optimizer = self.get_optimizer()
        self.optimizer = optimizer(self.model.parameters(), **optimizer_args)
        self.metrics = metrics

        # Search through metrics to find criterion index and set reference
        metrics_names = [metric.name for metric in metrics]
        criterion_index = metrics_names.index(criterion)
        self.criterion = self.metrics[criterion_index].value

    @staticmethod
    def get_optimizer():
        return torch.optim.Adam

    def forward(self, minibatch) -> ModelOutput:
        images = minibatch["features"]
        
        if torch.cuda.is_available():
            images = images.cuda()
        
        logits = self.model(images)
        loss = self.criterion(logits, images)
        
        return ModelOutput(loss, logits, images)

    def score(self, logits, labels, step: str):
        return {
            f"{metric.name}_{step}": metric.value(logits, labels).cpu().item()
            for metric in self.metrics
        }

    def training_step(self, minibatch, step="TRAIN") -> dict[str, float]:
        self.model.train()

        output = self.forward(minibatch)
        
        self.optimizer.zero_grad()
        output.loss.backward()
        self.optimizer.step()

        return self.score(output.logits, output.images, "TRAIN")

    @torch.no_grad()
    def validation_step(self, minibatch, step="VAL"):
        self.model.eval()
        output = self.forward(minibatch)
        return self.score(output.logits, output.images, step)

# COMMAND ----------


