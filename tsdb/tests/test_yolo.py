from typing import Any
from unittest.mock import Mock

import pytest
import torch

from ultralytics.utils import IterableSimpleNamespace

from tsdb.ml.yolo import YoloModelTrainer, ModifiedDetectionValidator

@pytest.fixture()
def lb() -> list[torch.Tensor]:
    return []


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
def training() -> bool:
    return False


@pytest.fixture()
def dataloader() -> Mock:
    return Mock()


@pytest.fixture()
def device() -> str:
    return "cpu"


@pytest.fixture()
def sample_batch():
    """
    A mock batch of data for the following tests.
    batch_idx corresponds to index of the image the bounding box is for in the im_file section of this dict.
    For example if batch_idx is 0 then the box is from the first image 'path/img1.jpg' in the im_file section.
    """
    shape = (2, 3, 1500, 1500)  # To create a random tensor (image) with the given shape
    batch = {
        "im_file": (
            "path/img1.jpg",
            "path/img2.jpg",
        ),
        "ori_shape": ((1500, 1500), (1500, 1500)),
        "resized_shape": ((1500, 1500), (1500, 1500)),
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


def test_get_validator(dataloader: Mock, training: bool, device: str, args: IterableSimpleNamespace):
    validator = YoloModelTrainer.get_validator(dataloader, training, device, args)
    assert isinstance(validator, ModifiedDetectionValidator), "Validator is not an instance of ModifiedDetectionValidator"


def test_label_loss_items():
    loss_types = ["box_loss", "BCE_loss", "DF_loss"]
    loss_items = torch.tensor([0.33, 0.12, 0.9])
    expected_labeled_loss_items = {"VAL/box_loss": 0.33, "VAL/BCE_loss": 0.12, "VAL/DF_loss": 0.9}
    labeled_loss_items = YoloModelTrainer.label_loss_items(loss_items, loss_types, "VAL")
    assert expected_labeled_loss_items == labeled_loss_items, "Labeled loss items are not as expected"
