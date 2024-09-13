# Databricks notebook source
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms, datasets
from PIL import Image
import os, glob, sys

# COMMAND ----------

# set registry to be UC en_model registry
mlflow.set_registry_uri("databricks-uc")

# create MLflow client
client = MlflowClient()

catalog = "edav_dev_csels"
schema = "towerscout_test_schema"

# COMMAND ----------

# MAGIC %run ./ts_en

# COMMAND ----------

model_weights_path = "/Volumes/edav_dev_csels/towerscout_test_schema/test_volume/model_params/en/b5_unweighted_best.pt"

en_model = EN_Classifier(model_weights_path)
