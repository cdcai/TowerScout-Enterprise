import pytest
import torch
from unittest.mock import MagicMock
from tsdb.ml.utils import OptimizerArgs
from tsdb.ml.yolo_trainer import YoloModelTrainer


@pytest.fixture
def mock_model():
    """
    Create a mock DetectionModel with necessary methods, attributes, and parameters.
    """
    model = MagicMock()
    model.args = MagicMock(half=False, amp=False)
    model.loss = MagicMock(
        return_value=(
            torch.tensor(1.0, requires_grad=True),  # Mocked loss with requires_grad=True
            [torch.tensor(0.5, requires_grad=True)] * 3,  # Mocked loss items
        )
    )
    model.forward = MagicMock(return_value=torch.randn(8, 10, requires_grad=True))  # Mocked predictions
    model.named_parameters.return_value = [
        ("layer1.weight", torch.nn.Parameter(torch.randn(10, 10, requires_grad=True))),
        ("layer1.bias", torch.nn.Parameter(torch.zeros(10, requires_grad=True))),
        ("bn1.weight", torch.nn.Parameter(torch.ones(10, requires_grad=True))),
        ("bn1.bias", torch.nn.Parameter(torch.zeros(10, requires_grad=True))),
    ]
    model.named_modules.return_value = [
        ("layer1", torch.nn.Linear(10, 10)),
        ("bn1", torch.nn.BatchNorm1d(10)),
    ]
    model.nc = 10  # Number of classes
    return model


def test_training_step(mock_model):
    """
    Test the training_step method of YoloModelTrainer.
    """
    optimizer_args = OptimizerArgs(optimizer_name="SGD", lr0=0.01, momentum=0.9)
    trainer = YoloModelTrainer(optimizer_args=optimizer_args, model=mock_model)

    # Mock input batch
    minibatch = {
        "img": torch.rand(8, 3, 640, 640, requires_grad=True),  # Mocked input images
        "batch_idx": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        "cls": torch.randint(0, 10, (8, 1)),  # Mocked class labels
        "bboxes": torch.rand(8, 4),  # Mocked bounding boxes
    }

    # Call the training_step method
    loss_scores = trainer.training_step(minibatch)

    # Assertions
    assert isinstance(loss_scores, dict), "Loss scores should be a dictionary."
    assert all(
        f"{loss_name}_TRAIN" in loss_scores for loss_name in trainer.loss_types
    ), "All loss types should be included in the output."
    assert trainer.loss.item() == 1.0, "Loss value should match the mocked value."

