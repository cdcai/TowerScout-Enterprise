# Databricks notebook source
!pip install efficientnet_pytorch

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from efficientnet_pytorch import EfficientNet
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

# set registry to be UC model registry
mlflow.set_registry_uri("databricks-uc")

# create MLflow client
client = MlflowClient()

logged_model_path = "/Volumes/edav_dev_csels/towerscout_test_schema/test_volume/model_params/en/b5_unweighted_best.pt"

# load original TowerScout Model weights
model = EfficientNet.from_pretrained("efficientnet-b5", include_top=True)
model._fc = nn.Sequential(nn.Linear(2048, 512), nn.Linear(512, 1))  # b5
model.cuda()
model._fc

checkpoint = torch.load(logged_model_path)
model.load_state_dict(checkpoint)
print("Loaded TowerScout weights")
run_name = "TowerScout_benchmark_model"

# COMMAND ----------

mlflow.autolog()
# test_img_path = "/Volumes/edav_dev_csels/towerscout_test_schema/test_volume/test_images/"
im_path = "/Volumes/edav_dev_csels/towerscout_test_schema/test_volume/test_images/tmp5j0qh5k10.jpg"
img = Image.open(im_path)
model.eval()

catalog = "edav_dev_csels"
schema = "towerscout_test_schema"
parent_dir = "/Workspace/Users/<user>"
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

mlflow.set_experiment("/Workspace/Users/<user>/TowerScout_YOLOv5_Baseline")

with mlflow.start_run(run_name="Register_pretrained_YOLOv5_model") as run:
    run_id = run.info.run_id

    y_pred = model.predict(img)

    signature = mlflow.infer_signature(img, y_pred)

    mlflow.pyfunc.log_model(model, run_name)


ts_baseline_model = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",  # path to logged artifact folder called models
    name=f"{catalog}.{schema}.{run_name}"
)



# COMMAND ----------


