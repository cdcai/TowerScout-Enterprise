from typing import Any
from unittest.mock import Mock

import pytest
import torch

import ultralytics
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.nn.tasks import DetectionModel

from tsdb.ml.model_trainer import BaseTrainer
from tsdb.ml.yolo import YoloModelTrainer, ModifiedDetectionValidator, YOLOLoss
from tsdb.ml.types import TrainingArgs


@pytest.fixture()
def training() -> bool:
    return False


@pytest.fixture()
def dataloader() -> Mock:
    return Mock()


@pytest.fixture()
def args() -> list[str]:
    """
    A mock args object for the following tests.
    We specify the threshold confidence score of 0.5 here 
    for testing the filtering feature of the postprocess function
    """
    return IterableSimpleNamespace(
        conf=0.5, iou=0.3, single_cls=True, max_det=300, save_hybrid=False
    )


@pytest.fixture()
def device() -> str:
    return "cpu"


@pytest.fixture()
def loss_items() -> torch.Tensor:
    return torch.tensor([0.33, 0.12, 0.9])


@pytest.fixture()
def loss_types() -> list[str]:
    return [loss.name for loss in YOLOLoss]


@pytest.fixture()
def step() -> str:
    return "VAL"


@pytest.fixture()
def expected_labeled_loss_items(step: str) -> dict[str, float]:
    return {f"{step}/box_loss": 0.33, f"{step}/BCE_loss": 0.12, f"{step}/DF_loss": 0.9}


@pytest.fixture()
def expected_labeled_loss_items_none(loss_types: list[str], step: str) -> dict[str, float]:
    return [f"{step}/{loss_type}" for loss_type in loss_types]


def test_get_validator(dataloader: Mock, training: bool, device: str, args: IterableSimpleNamespace):
    validator = YoloModelTrainer.get_validator(dataloader, training, device, args)
    assert isinstance(validator, ModifiedDetectionValidator), "Validator is not an instance of ModifiedDetectionValidator"


def test_label_loss_items(loss_items: torch.Tensor, expected_labeled_loss_items: dict[str, float], loss_types: list[str], step: str):
    labeled_loss_items = YoloModelTrainer.label_loss_items(loss_items, loss_types, step)
    assert expected_labeled_loss_items == labeled_loss_items, "Labeled loss items are not as expected"


def test_label_loss_items_none(expected_labeled_loss_items_none: dict[str, float], loss_types: list[str], step: str):
    """Test output of label_loss_items when loss_items is None"""
    labeled_loss_items = YoloModelTrainer.label_loss_items(None, loss_types, step)
    assert expected_labeled_loss_items_none == labeled_loss_items, "Labeled loss items are not as expected"


def test_get_model():
    model = YoloModelTrainer.get_model(f"yolov8n.yaml", "yolov8n.pt")
    assert isinstance(model, DetectionModel), "Model is not an instance of DetectionModel"


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
    return YoloModelTrainer.build_optimizer(model)


@pytest.fixture()
def trainer(model: DetectionModel, train_args: TrainingArgs, optimizer: torch.optim.Optimizer) -> BaseTrainer:
    trainer = YoloModelTrainer(model, train_args=train_args, optimizer=optimizer)
    return trainer


@pytest.fixture()
def sample_batch():
    """
    A mock batch of data for the following tests.
    batch_idx corresponds to index of the image the bounding box is for in the im_file section of this dict.
    For example if batch_idx is 0 then the box is from the first image 'path/img1.jpg' in the im_file section.
    """
    shape = (2, 3, 640, 640)  # To create a random tensor (image) with the given shape
    batch = {
        "im_file": (
            "path/img1.jpg",
            "path/img2.jpg",
        ),
        "ori_shape": ((640, 640), (640, 640)),
        "resized_shape": ((640, 640), (640, 640)),
        "ratio_pad": (None, None),
        "img": torch.randint(0, 256, shape, dtype=torch.uint8),
        "cls": torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0]]),  # class labels
        "bboxes": torch.tensor(
            [
                [0.5746, 0.6375, 0.2610, 0.3689],
                [0.3660, 0.6481, 0.1675, 0.3164],
                [0.5915, 0.5939, 0.1315, 0.1461],
                [0.4127, 0.5856, 0.1139, 0.1259],
                [0.3695, 0.7020, 0.0239, 0.0671],
            ]
        ),
        "batch_idx": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    }

    return batch


def test_training_step(trainer: YoloModelTrainer, sample_batch: dict[str, Any], loss_types: list[str]):
    expected_keys = [f"TRAIN/{loss_type}" for loss_type in loss_types] + ["loss"]
    loss_scores, loss_items = trainer.training_step(sample_batch)
    assert isinstance(loss_scores, dict), "loss_scores is not a dict"
    assert all(key in loss_scores for key in expected_keys), f"Missing key from: {expected_keys}"
    assert isinstance(loss_items, torch.Tensor), "loss_items is not a torch tensor"


def test_freeze_layers(trainer: YoloModelTrainer):
    """
    Testing freezing of layers. We freeze conv layers and the dfl layer of the DetectionModel.
    Prior to calling the function we unfreeze the layers we want the function to freeze and freeze
    all other to get full coverage of the function.
    """
    freeze_layer_names = [".dfl", "model.conv."]

    for k, v in trainer.model.named_parameters():
        if any(x in k for x in freeze_layer_names):   
            v.requires_grad = True
        else:
            v.requires_grad = False
    
    trainer.freeze_layers(["conv"]) 
    
    for k, v in trainer.model.named_parameters():
        if any(x in k for x in freeze_layer_names):
            assert v.requires_grad == False, "Not all specified layers are not frozen"
        else:
            assert v.requires_grad == True, "Layers that were not specified were frozen"



