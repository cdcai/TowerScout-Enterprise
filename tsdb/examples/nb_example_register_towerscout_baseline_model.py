# Databricks notebook source
!pip install efficientnet_pytorch
!pip install opencv-python # need for yolo model
!pip install ultralytics==8.2.92 # need for yolo model
!pip install gitpython==3.1.30 pillow==10.3.0 requests==2.32.0 setuptools==70.0.0 # need for loading yolo with torch.hub.load from ultralytics
!pip install shapely==2.0.3

# COMMAND ----------

# MAGIC %run ./ts_yolov5

# COMMAND ----------

from webapp.ts_en import EN_Classifier

# COMMAND ----------

# set registry to be UC model registry
mlflow.set_registry_uri("databricks-uc")

catalog = "edav_dev_csels"
schema = "towerscout"

# COMMAND ----------

en_model_weight_path = f"/Volumes/{catalog}/{schema}/misc/model_params/en/b5_unweighted_best.pt"
en_model = EN_Classifier(en_model_weight_path)

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
import torch
from PIL import Image
import os, glob, sys

# COMMAND ----------

yolo_dep_path = (
    f"/Volumes/{catalog}/{schema}/misc/yolov5"
)

model_weights_path = f"/Volumes/{catalog}/{schema}/misc/model_params/yolo/xl_250_best.pt"

# load YOLOv5 model
model = torch.hub.load(
    repo_or_dir=yolo_dep_path, model="custom", path=model_weights_path, source="local"
)

# COMMAND ----------

# create YOLOv5_Detector Mlflow PythonModel
yolo_model = YOLOv5_Detector(model=model, batch_size=1)

# COMMAND ----------

img_path = f"/Volumes/{catalog}/{schema}/misc/test_images/"

png_files = glob.glob(os.path.join(img_path, "*.png"))

# get 5 test images as np arrays
x_test = np.array(
    [np.asarray(Image.open(png_files[i]), dtype=np.float32) for i in range(5)]
)

# COMMAND ----------

# we will pass a list of Image objects or np arrays instead of the images transformed into tensors
# or else we get an error where .xyxyn is not avaiable in the output of the model

with mlflow.start_run() as run:
    run_id = run.info.run_id

    yolo_model.model.eval()

    y_pred = yolo_model.predict(context=None, model_input=x_test, secondary=en_model)

    # hard coding model_output because current test images yield no detection output
    yolo_sig = infer_signature(
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

    en_sig = infer_signature(
        model_input=x_test,
        model_output=0.0
    )

    # log model as custom model using pyfunc flavor. Note that the model must inheret from PythonModel Mlflow class
    mlflow.pyfunc.log_model(
        python_model=yolo_model,
        artifact_path="base_yolov5_model",
        signature=yolo_sig,
        pip_requirements=[
            "ultralytics==8.2.92",
            "gitpython==3.1.30",
            "pillow==10.3.0",
            "requests==2.32.0",
            "setuptools==70.0.0",
        ],
    )
    mlflow.pytorch.log_model(
        pytorch_model=en_model,
        artifact_path="base_EN_model",
        signature=en_sig,
    )
    

# COMMAND ----------

# DBTITLE 1,Register baseline YOLO model
model_name = f"{catalog}.{schema}.baseline"  # will be model name in UC

registered_yolo_model_metadata = yolo_model.register_model(
    model_name, run_id, "base_yolov5_model"
)

# COMMAND ----------

# DBTITLE 1,Register base EN model
registered_en_model_metadata = mlflow.register_model(
    model_uri=f"runs:/{run_id}/base_EN_model",
    name=f"{catalog}.{schema}.base_en",
)

# COMMAND ----------

### Test retrieval
# Identify test image
img_path = f"/Volumes/{catalog}/{schema}/misc/test_images/"

png_files = glob.glob(os.path.join(img_path, "*.png"))

# get 5 test images as np arrays
x_test = np.array(
    [np.asarray(Image.open(png_files[i]), dtype=np.float32) for i in range(5)])

# COMMAND ----------

# Retrieve models
registered_yolo_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/11"
)

registered_en_model = mlflow.pyfunc.load_model(
    model_uri=f"runs:/{run_id}/base_EN_model")

# COMMAND ----------

y_pred = registered_yolo_model.predict(x_test)
print(y_pred)
