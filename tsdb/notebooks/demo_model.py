# Databricks notebook source
import torch
from torch import nn
from enum import Enum
from collections import namedtuple

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
    def __init__(self, optimizer=None, metrics=None, criterion: str="MSE"):
        self.model = Autoencoder()

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        if metrics is None:
            metrics = {
                "MSE": DemoMetrics.MSE.value,
            }
        
        self.optimizer = optimizer
        self.metrics = metrics
        self.criterion = "MSE"
    
    def forward(self, minibatch) -> ModelOutput:
        images = minibatch["features"]
        
        if torch.cuda.is_available():
            images = images.cuda()
        
        logits = self.model(images)
        loss = self.metrics[self.criterion](logits, images)
        
        return ModelOutput(loss, logits, images)

    def score(self, logits, labels, step: str):
        return {
            f"{metric}/{step}": metric_func(logits, labels).cpu().item()
            for metric, metric_func in self.metrics.items()
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
