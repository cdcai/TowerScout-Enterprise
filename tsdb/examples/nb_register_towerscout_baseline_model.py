# Databricks notebook source
!pip install efficientnet_pytorch

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec
from mlflow import MlflowClient
from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms, datasets
from PIL import Image
import sys, os

sys.path.append(
    "/Workspace/Repos/DDPHSS-CSELS-PD-TOWERSCOUT/TowerScout/Models/baseline"
)
sys.path.append(
    "/Workspace/Repos/DDPHSS-CSELS-PD-TOWERSCOUT/TowerScout/webapp/ts_yolov5/"
)
# from ts_yolov5 import YOLOv5_Detector
print(sys.path)

# COMMAND ----------



# set registry to be UC en_model registry
mlflow.set_registry_uri("databricks-uc")

# create MLflow client
client = MlflowClient()

logged_en_model_path = "/Volumes/edav_dev_csels/towerscout_test_schema/test_volume/en_model_params/en/b5_unweighted_best.pt"
logged_yolo_path = "/Volumes/edav_dev_csels/towerscout_test_schema/test_volume/en_model_params/yolo/xl_250_best.pt"

# load original TowerScout EfficientNet Model weights
en_model = EfficientNet.from_pretrained("efficientnet-b5", include_top=True)
en_model._fc = nn.Sequential(nn.Linear(2048, 512), nn.Linear(512, 1))  # b5
if torch.cuda.is_available():
    en_model.cuda()
    # en_model._fc    
    checkpoint = torch.load(logged_en_model_path)
else:
    checkpoint = torch.load(logged_en_model_path, map_location=torch.device("cpu"))

en_model.load_state_dict(checkpoint)
print("Loaded TowerScout EffificentNet weights")

# load YOLO v5 Model Weights
yolo_model = YOLOv5_Detector(logged_yolo_path)
print("Loaded TowerScout YOLO Weight")

run_name = "TowerScout_benchmark_en_model"

# COMMAND ----------

mlflow.autolog()
# test_img_path = "/Volumes/edav_dev_csels/towerscout_test_schema/test_volume/test_images/"
im_path = "/Volumes/edav_dev_csels/towerscout_test_schema/test_volume/test_images/tmp5j0qh5k10.jpg"
img = Image.open(im_path)
en_model.eval()
yolo_model.eval()

catalog = "edav_dev_csels"
schema = "towerscout_test_schema"
# parent_dir = "/Workspace/Users/nzs0@cdc.gov"
# if not os.path.exists(parent_dir):
#     os.makedirs(parent_dir)

# mlflow.set_experiment("/Workspace/Users/nzs0@cdc.gov/TowerScout_YOLOv5_Baseline")
mlflow.pytorch.autolog(disable=False)

transform = transforms.ToTensor()

with mlflow.start_run(run_name="Register_pretrained_YOLOv5_model_2") as run:
    run_id = run.info.run_id
    x_test = transform(img).unsqueeze(0)
    y_pred = yolo_model.detect(x_test, secondary=en_model)
    # y_pred = model.forward(x_test)

    input_schema = Schema([TensorSpec(np.dtype("float32"), x_test.shape)])
    output_schema = Schema([TensorSpec(np.dtype("float32"), y_pred.shape)])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    mlflow.pytorch.log_model(yolo_model, run_name)
    # mlflow.pytorch.log_model(en_model, run_name)
    
    client.set_tag(run_id=run_id, key="model_type", value="baseline_yoloV5")


ts_baseline_model = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",  # path to logged artifact folder called models
    name=f"{catalog}.{schema}.{run_name}"
)



# COMMAND ----------


