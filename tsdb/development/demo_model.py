# Databricks notebook source
import torch
from torch import nn, Tensor
from enum import Enum, auto

from collections import namedtuple
from functools import partial

# COMMAND ----------

class DemoMetrics(Enum):
    MSE = nn.MSELoss()

class DemoSteps(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()

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

def score_demo(logits, labels, step: str, metrics):
        return {
            f"{metric.name}_{step}": metric.value(logits, labels).cpu().item()
            for metric in metrics
        }

def forward_func_demo(model, minibatch) -> tuple[Tensor,Tensor,Tensor]:
        images = minibatch["features"]

        if torch.cuda.is_available():
            images = images.cuda()

        logits = model(images)
        # labels are image features 
        return logits, images, images
    

@torch.no_grad()
def inference_step_demo(minibatch, model, metrics, step) -> dict:
    model.eval()
    logits, _, labels = forward_func_demo(model, minibatch)
    return score_demo(logits, labels, step, metrics)

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

    # def forward(self, minibatch) -> ModelOutput:
    #     images = minibatch["features"]
        
    #     if torch.cuda.is_available():
    #         images = images.cuda()
        
    #     logits = self.model(images)
    #     loss = self.criterion(logits, images)
        
    #     return ModelOutput(loss, logits, images)

    def training_step(self, minibatch) -> dict[str, float]:
        self.model.train()

        logits, images, labels = forward_func_demo(self.model, minibatch)
        loss = self.criterion(logits, images)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return score_demo(logits, labels, DemoSteps["TRAIN"].name, self.metrics)

    @torch.no_grad()
    def validation_step(self, minibatch):
        return inference_step_demo(minibatch, self.model, self.metrics, DemoSteps["VAL"].name)
        # self.model.eval()
        # output = self.forward(minibatch)
        # return self.score(output.logits, output.images, step)
