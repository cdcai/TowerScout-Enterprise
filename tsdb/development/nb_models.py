# Databricks notebook source
import torch
from typing import Protocol
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, FloatType
from torch import Tensor
from torch import nn
import sys


def instantiate_inference_model(model: nn.Module) -> nn.Module:
    # Happens in memory but double check
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    return model


class InferenceModelType(Protocol):
    """
    A model class to wrap the model and provide the required methods to run
    distributed inference
    TODO: model instantiation logic should be moved to the model class
    """

    @property
    def model(self) -> nn.Module:
        raise NotImplementedError

    @property
    def return_type(self) -> StructType:
        raise NotImplementedError

    def __call__(self, input) -> Tensor:  # dunder methods
        raise NotImplementedError

    def preprocess_input(self, input) -> Tensor:
        # torchvision.transforms
        raise NotImplementedError
