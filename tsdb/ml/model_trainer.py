import torch
from torch import nn, optim
from torch import Tensor
from torchvision import transforms, datasets
from efficientnet_pytorch import EfficientNet
from enum import Enum, auto
from collections import namedtuple

from tsdb.ml.models import TowerScoutModel
from tsdb.ml.data_processing import transform_row, get_transform_spec, get_converter 


class Metrics(Enum):
    MSE = nn.MSELoss()

class Steps(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


def set_optimizer(model, optlr=0.0001, optmomentum=0.9, optweight_decay=1e-4):
    params_to_update = []
    for name, param in model.named_parameters():
        if "_bn" in name:
            param.requires_grad = False
        if param.requires_grad == True:
            params_to_update.append(param)

    optimizer = optim.SGD(
        params_to_update, lr=optlr, momentum=optmomentum, weight_decay=optweight_decay
    )
    return optimizer


def score(logits, labels, step: str, metrics: Metrics):
        return {
            f"{metric.name}_{step}": metric.value(logits, labels).cpu().item()
            for metric in metrics
        }

def forward_func(model, minibatch) -> tuple[Tensor,Tensor,Tensor]:
        images = minibatch["features"]
        labels = minibatch["labels"]

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        logits = model(images)

        return logits, images, labels
    

@torch.no_grad()
def inference_step(minibatch, model, metrics, step) -> dict:
    model.eval()
    logits, _, labels = forward_func(model, minibatch)
    return score(logits, labels, step, metrics)


class TowerScoutModelTrainer:
    def __init__(self, optimizer_args, metrics=None, criterion: str = "MSE"):
        self.model = TowerScoutModel()

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

    @staticmethod
    def get_optimizer():
        return torch.optim.Adam

    def training_step(self, minibatch, **kwargs) -> dict:
        self.model.train()

        logits, images, labels = forward_func(self.model, minibatch)
        loss = self.criterion(logits, images)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return score(logits, labels, Steps["TRAIN"].name, self.metrics)

    @torch.no_grad()
    def validation_step(self, minibatch, **kwargs) -> dict:
        return inference_step(minibatch, self.model, self.metrics, Steps["VAL"].name)

    def save_model(self):
        pass


