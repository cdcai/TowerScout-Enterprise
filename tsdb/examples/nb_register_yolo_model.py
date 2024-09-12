# Databricks notebook source
!pip install opencv-python # need for yolo model

# COMMAND ----------

# MAGIC %run ./ts_yolov5

# COMMAND ----------

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

filename = "/Volumes/edav_dev_csels/towerscout_test_schema/test_volume/model_params/yolo/xl_250_best.pt"

# load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path=filename)

# create MLflow yolov5 model
yolo_model = YOLOv5_Detector(model=model, batch_size=1)

# COMMAND ----------

img_path = "/Volumes/edav_dev_csels/towerscout_test_schema/test_volume/test_images/"

jpg_files = glob.glob(os.path.join(img_path, "*.jpg"))

# get 5 test images as np arrays
# must be a np array of np arrays not list of np arrays or else signature enforcement breaks...
x_test = np.array(
    [np.asarray(Image.open(jpg_files[i]), dtype=np.float32) for i in range(5)]
)

# COMMAND ----------

# we will pass a list of Image objects or np arrays instead of the images transformed into tensors or else we get an error where .xyxyn is avaiable in the output of the model

with mlflow.start_run() as run:
    run_id = run.info.run_id

    yolo_model.model.eval()

    y_pred = yolo_model.predict(context=None, model_input=x_test)

    # hard coding model_output because model currently doesnt detect stuff
    sig = infer_signature(
        model_input=x_test,
        model_output=[
            {
                "x1": 0.0,
                "y1": 0.0,
                "x2": 0.0,
                "y2": 0.0,
                "conf": 0.0,
                "class": 0,
                "class_name": "",
                "secondary": 1,
            }
        ],
    )

    # log model as custom model using pyfunc flavor. Note that the model must inheret from PythonModel Mlflow class
    mlflow.pyfunc.log_model(
        python_model=yolo_model,
        artifact_path="base_yolov5_model",
        signature=sig,
        pip_requirements=[
            "ultralytics==8.2.92",
        ],
    )

# COMMAND ----------

# DBTITLE 1,Register model
model_name = "towerscout_baseline_model"

ts_baseline_model_metadata = mlflow.register_model(
    model_uri=f"runs:/{run_id}/base_yolov5_model",  # path to logged artifact folder called models
    name=f"{catalog}.{schema}.{model_name}",
)

# COMMAND ----------

# DBTITLE 1,Set registered model alias
alias = "testing"
client.set_registered_model_alias(
    name=ts_baseline_model_metadata.name,
    alias=alias,
    version=ts_baseline_model_metadata.version,  # get version of model from when it was registered
)

# COMMAND ----------

model_name = f"{catalog}.{schema}.towerscout_baseline_model"  # model name in UC
# get requierments.txt file to install dependecies for yolov5 module i.e. ultralyitcs
path_to_req_txt = mlflow.pyfunc.get_model_dependencies(f"models:/{model_name}@{alias}")

# COMMAND ----------

# MAGIC %pip install -r {path_to_req_txt}

# COMMAND ----------

# DBTITLE 1,Load registred model for inference
# make this attroibute of detecotr class and append path to sys in the __init__ method
yolo_dep_path = "/Volumes/edav_dev_csels/towerscout_test_schema/ultralytics_yolov5_master"

# IMPORTANT when loading the model you must append the path to this directory to the system path so
# Python looks there for the files/modules needed to load the yolov5 module
sys.path.append(yolo_dep_path)

model_name = f"{catalog}.{schema}.towerscout_baseline_model"  # model name in UC

registered_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}@{alias}"
)

# COMMAND ----------

x_test = np.array(
    [np.asarray(Image.open(jpg_files[i]), dtype=np.float32) for i in range(5, 10)]
)

y_pred = registered_model.predict(x_test)  # perform inference

print(f"Inference results for 5 images: {y_pred}")
