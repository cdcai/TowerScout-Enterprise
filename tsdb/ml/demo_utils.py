import torch
from torch import nn, Tensor
from enum import Enum, auto

from collections import namedtuple
from functools import partial

from tsdb.ml.models import Autoencoder


class DemoMetrics(Enum):
    MSE = nn.MSELoss()

class DemoSteps(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


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

