# Databricks notebook source
!pip install efficientnet_pytorch
!pip install shapely==2.0.3
!pip install opencv-python # need for yolo model
!pip install mlflow-skinny==2.15.1
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from mlflow.models.signature import ModelSignature 
from mlflow.types import Schema, ColSpec, DataType, ParamSchema, ParamSpec
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
import pandas as pd
import os, glob, sys
import cv2 

from webapp.ts_en import EN_Classifier
from tsdb.utils.uc import CatalogInfo

# COMMAND ----------

# set registry to be UC model registry
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

catalog_info = CatalogInfo.from_spark_config(spark)
catalog = catalog_info.name
schema = "towerscout"

# COMMAND ----------

alias = 'aws'
en_model_name = f"{catalog}.{schema}.efficientnet"
yolo_model_name = f"{catalog}.{schema}.yolo_autoshape"


registered_en_model = mlflow.pytorch.load_model(
    model_uri=f"models:/{en_model_name}@{alias}"
    )

yolo_detector = YOLOv5_Detector_new.from_uc_registry(model_name=yolo_model_name, alias='aws', batch_size=1)
yolo_detector.model.eval()

print(f"Yolo model in eval mode? {not yolo_detector.model.training}")
print(f"EN model in eval mode? {not registered_en_model.training}")

# COMMAND ----------

jpg_path = f"/Volumes/edav_dev_csels/towerscout/misc/bronze_sample/"
png_path = f"/Volumes/edav_dev_csels/towerscout/data/Training Data/nyc/"

def get_imgs(path, use_pil=True, num_imgs=10):
    files = glob.glob(os.path.join(path, "*"))

    if use_pil:
        x = [Image.open(files[i]) for i in range(num_imgs)]
    else:
        x = (cv2.imread(files[i], cv2.IMREAD_COLOR) for i in range(num_imgs))
        x = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in x]
    return x


use_pil = True
num_imgs = 20

x_png = get_imgs(png_path, use_pil, num_imgs)
x_jpg = get_imgs(jpg_path, use_pil, num_imgs)

x_test = x_jpg

# COMMAND ----------

transform = transforms.Compose([
            transforms.Resize([456, 456]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5553, 0.5080, 0.4960), std=(0.1844, 0.1982, 0.2017))
            ])

files = glob.glob(os.path.join(jpg_path, "*"))

for i in range(len(x_test)):
    try:
        img_t = transform(x_test[i])
        print(f"Good img:", files[i])
    except:
        print(f"Error img:", files[i])

# COMMAND ----------

yolo_detector.batch_size = 32
y_pred = yolo_detector.predict(model_input=x_test, secondary=None)
print(y_pred)

# COMMAND ----------

y_pred = yolo_detector.predict(model_input=x_test, secondary=registered_en_model)
print(y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC Run 1:
# MAGIC
# MAGIC RuntimeError: output with shape [1, 456, 456] doesn't match the broadcast shape [3, 456, 456]
# MAGIC
# MAGIC Run 2:
# MAGIC
# MAGIC RuntimeError: output with shape [1, 456, 456] doesn't match the broadcast shape [3, 456, 456]
# MAGIC
# MAGIC Run 3:
# MAGIC
# MAGIC RuntimeError: output with shape [1, 456, 456] doesn't match the broadcast shape [3, 456, 456]
