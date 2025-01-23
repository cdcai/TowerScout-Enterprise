from typing import Any
from unittest.mock import Mock

import pytest
import torch

import ultralytics
from ultralytics.nn.tasks import DetectionModel

from tsdb.ml.model_trainer import BaseTrainer
from tsdb.ml.types import TrainingArgs


@pytest.fixture()
def train_args() -> TrainingArgs:
    return TrainingArgs()


@pytest.fixture()
def model() -> DetectionModel:
    model = DetectionModel(verbose=False)
    model.args = ultralytics.cfg.get_cfg()
    return model


@pytest.fixture()
def optimizer(model: DetectionModel) -> torch.optim.Optimizer:
    return BaseTrainer.build_optimizer(model)


@pytest.fixture()
def trainer(model: DetectionModel, train_args: TrainingArgs, optimizer: torch.optim.Optimizer) -> BaseTrainer:
    trainer = BaseTrainer(model, train_args=train_args, optimizer=optimizer)
    return trainer


def test_build_optimizer(trainer: BaseTrainer) -> None:
    optimizer = BaseTrainer.build_optimizer(trainer.model, name="auto")
    assert isinstance(optimizer, torch.optim.Optimizer), "optimizer is not an instance of torch.optim.Optimizer"

    optimizer = BaseTrainer.build_optimizer(trainer.model, name="Adam")
    assert isinstance(optimizer, torch.optim.Adam), "optimizer is not an instance of torch.optim.Adam"

    optimizer = BaseTrainer.build_optimizer(trainer.model, name="RMSProp")
    assert isinstance(optimizer, torch.optim.RMSprop), "optimizer is not an instance of torch.optim.RMSprop"

    optimizer = BaseTrainer.build_optimizer(trainer.model, name="SGD")
    assert isinstance(optimizer, torch.optim.SGD), "optimizer is not an instance of torch.optim.SGD"

    with pytest.raises(NotImplementedError):
        # should raise NotImplementedError exception
        optimizer = BaseTrainer.build_optimizer(trainer.model, name="Unknown")


def test__setup_scheduler(trainer: BaseTrainer) -> None:
    """ 
    Test _setup_scheduler method with both cos_lr being true and false
    """
    trainer.train_args.cos_lr = False
    trainer._setup_scheduler()
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.LRScheduler), "scheduler is not an instance of torch.optim.lr_scheduler.LRScheduler"
    
    trainer.train_args.cos_lr = True
    trainer._setup_scheduler()
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.LRScheduler), "scheduler is not an instance of torch.optim.lr_scheduler.LRScheduler"

