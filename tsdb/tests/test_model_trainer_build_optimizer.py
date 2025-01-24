import pytest
import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW, RMSprop
from tsdb.ml.utils import OptimizerArgs
from tsdb.ml.yolo_trainer import YoloModelTrainer
from unittest.mock import MagicMock 

@pytest.fixture(scope="module")
def mock_model():
    model = MagicMock()
    model.nc = 10
    model.named_modules.return_value = [
        ("conv1", nn.Conv2d(3, 16, kernel_size=3)),  # Regular weights -> g0
        ("bn1", nn.BatchNorm2d(16)),                # BatchNorm weights -> g1
    ]
    model.named_parameters.return_value = [
        ("conv1.weight", nn.Parameter(torch.empty(16, 3, 3, 3))),  # Regular weights -> g0
        ("conv1.bias", nn.Parameter(torch.empty(16))),             # Bias -> g2
        ("bn1.weight", nn.Parameter(torch.empty(16))),             # BatchNorm weights -> g1
        ("bn1.bias", nn.Parameter(torch.empty(16))),               # BatchNorm bias -> g2
    ]
    return model




@pytest.mark.parametrize(
    "optimizer_name, iterations, expected_optimizer, expected_lr",
    [
        ("auto", 1e6, SGD, 0.01),        # High iterations -> SGD
        ("auto", 1e3, AdamW, 0.000714),  # Low iterations -> AdamW
        ("Adam", None, Adam, 0.001),     # Explicit Adam
        ("RMSProp", None, RMSprop, 0.001), # Explicit RMSProp
    ],
)

def test_build_optimizer(mock_model, optimizer_name, iterations, expected_optimizer, expected_lr):
    optimizer_args = OptimizerArgs(
        optimizer_name=optimizer_name,
        lr0=0.001,
        momentum=0.9,
    )
    trainer = YoloModelTrainer(optimizer_args=optimizer_args, model=mock_model)
    optimizer = trainer.build_optimizer(mock_model, name=optimizer_name, iterations=iterations or 1e5)
    assert isinstance(optimizer, expected_optimizer), (
        f"Expected {expected_optimizer.__name__}, but got {type(optimizer).__name__}"
    )
    assert optimizer.defaults["lr"] == expected_lr, f"Expected learning rate {expected_lr}, but got {optimizer.defaults['lr']}."

def test_build_optimizer_invalid_optimizer(mock_model):
    """
    Test that the build_optimizer method raises an error for an invalid optimizer name.
    """
    # Create a mock OptimizerArgs object
    optimizer_args = OptimizerArgs(
        optimizer_name="SGD",  # Provide a valid optimizer initially
        lr0=0.001,
        momentum=0.9,
    )
    
    # Initialize YoloModelTrainer with valid arguments
    trainer = YoloModelTrainer(optimizer_args=optimizer_args, model=mock_model)
    
    # Call build_optimizer with an invalid optimizer name and catch the error
    try:
        trainer.build_optimizer(mock_model, name="InvalidOptimizer")
    except NotImplementedError:
        # Test passes if NotImplementedError is raised
        pass
    else:
        # Fail the test if no error is raised
        pytest.fail("Expected NotImplementedError was not raised.")

def test_loss_types(mock_model):
    """
    Test that the trainer includes all defined YOLO loss types.
    """
    optimizer_args = OptimizerArgs(optimizer_name="SGD", lr0=0.01, momentum=0.9)
    trainer = YoloModelTrainer(optimizer_args=optimizer_args, model=mock_model)

    expected_loss_types = ["BL", "BCE", "DLF"]
    assert trainer.loss_types == expected_loss_types, f"Expected loss types {expected_loss_types}, but got {trainer.loss_types}."

def test_model_device_placement(mock_model):
    """
    Test that the model is correctly moved to the available device.
    """
    optimizer_args = OptimizerArgs(optimizer_name="SGD", lr0=0.01, momentum=0.9)
    trainer = YoloModelTrainer(optimizer_args=optimizer_args, model=mock_model)

    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert trainer.device == expected_device, f"Expected device {expected_device}, but got {trainer.device}."

