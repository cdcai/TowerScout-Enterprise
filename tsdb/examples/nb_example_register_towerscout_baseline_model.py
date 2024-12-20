# Databricks notebook source
!pip install efficientnet_pytorch
!pip install opencv-python # need for yolo model
!pip install ultralytics==8.2.92 # need for yolo model
!pip install gitpython==3.1.30 pillow==10.3.0 requests==2.32.0 setuptools==70.0.0 # need for loading yolo with torch.hub.load from ultralytics
!pip install shapely==2.0.3
!pip install mlflow-skinny==2.15.1 # need this if on GPU 
dbutils.library.restartPython() # need this if on GPU

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from mlflow.models.signature import ModelSignature 
from mlflow.types import Schema, ColSpec, DataType
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch
from PIL import Image
import cv2 
import os, glob

from tsdb.utils.uc import CatalogInfo
from tsdb.ml.detections import YOLOv5_Detector
from tsdb.ml.efficientnet import ENClassifier

# COMMAND ----------

# set registry to be UC model registry
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

catalog_info = CatalogInfo.from_spark_config(spark)
catalog = catalog_info.name
schema = "towerscout"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initializing model objects to perform initial model logging and registration 
# MAGIC If you intend to fine tune the EN model you may need a 
# MAGIC proxy connection which may not be available in production.

# COMMAND ----------

# DBTITLE 1,Set up AWS EN model
en_model_weight_path = f"/Volumes/{catalog}/{schema}/misc/model_params/en/b5_unweighted_best.pt"

# Use from_name instead of from_pretrained to avoid downloading pretrained weights
en_model = EfficientNet.from_name(model_name='efficientnet-b5', include_top=True)

en_model._fc = nn.Sequential(
            nn.Linear(2048, 512), #b5
            nn.Linear(512, 1)
        )

if torch.cuda.is_available():
    checkpoint = torch.load(en_model_weight_path)
else:
    checkpoint = torch.load(en_model_weight_path, map_location=torch.device('cpu'))

en_model.load_state_dict(checkpoint)
en_model.eval()

# COMMAND ----------

# DBTITLE 1,Set up AWS YOLO model
yolo_dep_path = (
    f"/Volumes/{catalog}/{schema}/misc/yolov5"
)

model_weights_path = f"/Volumes/{catalog}/{schema}/misc/model_params/yolo/xl_250_best.pt"

# load YOLOv5 model (Autoshape object which inherits from nn.Module)
yolo_model = torch.hub.load(
    repo_or_dir=yolo_dep_path, model="custom", path=model_weights_path, source="local"
)

yolo_model.eval()

# COMMAND ----------

# we will pass a list of Image objects instead of the images transformed into tensors
# or else we get an error where .xyxyn is not avaiable in the output of the model

with mlflow.start_run() as run:
    run_id = run.info.run_id

    # using DataType.binary as the input schema type allows us to use any python object
    # but it seems the only way to get this to work is to make the input data a 
    # pd dataframe with a column named "images" and the value is the image data...
    input_schema = Schema([ColSpec(DataType.binary, name="images")]) # Create the signature
    output_schema = Schema([ColSpec(DataType.binary, name="outputs")]) # Create the signature

    yolo_sig = ModelSignature(inputs=input_schema, outputs=output_schema)
    en_sig = ModelSignature(inputs=input_schema, outputs=output_schema)

    # log model as custom model using pytorch flavor. Note that the model must inheret from PythonModel Mlflow class
    mlflow.pytorch.log_model(
        pytorch_model=yolo_model,
        artifact_path="yolo_autoshape",
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
        artifact_path="efficientnet",
        signature=en_sig,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Registering models
# MAGIC
# MAGIC Note that we give the models the alias's baseline and aws. The alias aws indicates that these models use the weights used by the models currently in use by towerscout in AWS. The alias baseline will indicate that these models are currently the models being used in production by towerscout. Lastly the tag yolo_version for the YOLO model is meant to indicate which yolo version we are using. In this case the model in AWS is using version 5 so we set the tag value to "v5".

# COMMAND ----------

# DBTITLE 1,Register AWS YOLO model
yolo_model_name = f"{catalog}.{schema}.yolo_autoshape"  # will be model name in UC

registered_yolo_model_metadata = mlflow.register_model(
    model_uri=f"runs:/{run_id}/yolo_autoshape",
    name=yolo_model_name,
)


aliases = ["baseline", "aws"]

for alias in aliases:
    client.set_registered_model_alias(
        name=yolo_model_name, alias=alias, version=registered_yolo_model_metadata.version
    )

# set yolo version as a tag
client.set_model_version_tag(
    name=yolo_model_name,
    version=registered_yolo_model_metadata.version,
    key="yolo_version",
    value="v5",
)

# COMMAND ----------

# DBTITLE 1,Register AWS EN model
en_model_name = f"{catalog}.{schema}.efficientnet"

registered_en_model_metadata = mlflow.register_model(
    model_uri=f"runs:/{run_id}/efficientnet",
    name=en_model_name,
)

for alias in aliases:
    client.set_registered_model_alias(
        name=en_model_name, alias=alias, version=registered_en_model_metadata.version
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load models for inference use

# COMMAND ----------

alias = "aws"
yolo_model_name = f"{catalog}.{schema}.yolo_autoshape" 
en_model_name = f"{catalog}.{schema}.efficientnet"  

# Retrieves models by alias and create inference objects
yolo_detector = YOLOv5_Detector.from_uc_registry(model_name=yolo_model_name, alias=alias, batch_size=16)
en_classifier = ENClassifier.from_uc_registry(model_name=en_model_name, alias=alias)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Perform inference on a sample of bronze images

# COMMAND ----------

jpg_path = f"/Volumes/edav_dev_csels/towerscout/misc/bronze_sample/"
png_path = f"/Volumes/edav_dev_csels/towerscout/data/Training Data/nyc/"

def get_imgs(path, use_pil=True, num_imgs=10):
    files = glob.glob(os.path.join(path, "*"))

    if use_pil:
        x = [Image.open(files[i]).convert("RGB") for i in range(num_imgs)]
    else:
        x = (cv2.imread(files[i], cv2.IMREAD_COLOR) for i in range(num_imgs))
        x = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in x]
    return x


# ultralytics lib accepts this (see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/model.py) and so does the yolov5 lib (see https://github.com/ultralytics/yolov5/blob/master/models/common.py) 
# No need to resize for yolov5 lib (see letterbox and exif_transpose funcs in https://github.com/ultralytics/yolov5/blob/master/models/common.py) as it does it for you 
# however the ultralytics lib (see load_image func in BaseDataset class in https://github.com/ultralytics/ultralytics/blob/591fdbd8b1a48eb820bd6dffe3d128db809f323d/ultralytics/data/base.py#L168) seems to have the resizing done in the dataset class not the models forward pass...
# get some test images as PIL Image objects.
use_pil = True
num_imgs = 10

x_png = get_imgs(png_path, use_pil, num_imgs)
x_jpg = get_imgs(jpg_path, use_pil, num_imgs)

x_test = x_jpg
y_pred = yolo_detector.predict(model_input=x_test, secondary=en_classifier)
print(y_pred)

# COMMAND ----------

y_pred[0]

# COMMAND ----------

import pandas as pd

df = pd.DataFrame(y_pred, columns=["predictions"])
display(df)

# COMMAND ----------


