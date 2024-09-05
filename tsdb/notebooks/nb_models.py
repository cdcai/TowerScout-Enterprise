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

def get_yolo_model_files(catalog: str, schema: str, cwd: str) -> None:
    """
    Used to move the ultralytics_yolov5_master folder from volumes to the current working directory as 
    this contains the files needed to run the YOLOv5 model
    """
    source_volume_path = f"dbfs:/Volumes/{catalog}/{schema}/ultralytics_yolov5_master/"
    target_workspace_path = f"file:{cwd}/ultralytics_yolov5_master/"
    dbutils.fs.cp(source_volume_path, target_workspace_path, recurse=True)
    repo_or_dir = f"{cwd}/ultralytics_yolov5_master/"
    
    # Append to path to let workers have access to these files
    sys.path.append(repo_or_dir)

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

    def __call__(self, input) -> Tensor: # dunder methods
        raise NotImplementedError

    def preprocess_input(self, input) -> Tensor:
        # torchvision.transforms
        raise NotImplementedError
