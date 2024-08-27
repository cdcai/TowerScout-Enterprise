# Databricks notebook source
import mlflow

from typing import Any

import torch
from torch import nn

import pandas as pd

from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, FloatType

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import sys

# COMMAND ----------

# MAGIC %pip install efficientnet_pytorch
# MAGIC %pip install opencv-python

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %run ./dataloader_development

# COMMAND ----------

dbutils.widgets.text("source_schema", defaultValue="towerscout_test_schema")
dbutils.widgets.text("source_table", defaultValue="image_metadata")
dbutils.widgets.text("batch_size", defaultValue="5")

# COMMAND ----------

# set registry to be UC model registry
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

catalog_info = CatalogInfo.from_spark_config(
    spark
)  # CatalogInfo class defined in utils nb

catalog = catalog_info.name
schema = dbutils.widgets.get("source_schema")
source_table = dbutils.widgets.get("source_table")

table_name = f"{catalog}.{schema}.{source_table}"

images = spark.read.format("delta").table(table_name).select("content", "path")
#images = spark.readStream.format("delta").table(table_name).select("content", "path")

# COMMAND ----------

# List of all registered models in UC
# Note that: Argument 'filter_string' is unsupported for models in the Unity Catalog.
registered_models = mlflow.search_registered_models()

# Filter models by catalog and schema
ts_models = []

for model in registered_models:
    catalog_name, schema_name, name = model.name.split(".")
    if catalog_name == catalog and schema_name == schema:
        ts_models.append(model.name)

dbutils.widgets.dropdown("model", ts_models[0], ts_models)


# COMMAND ----------

class ModelReturnTypes(Enum):
    Autoencoder = StructType([StructField("logits", ArrayType(ArrayType(FloatType())), True)])
    YOLO = StructType([StructField("bbox", ArrayType(FloatType()), True)])
    EfficientNet = StructType([StructField("score", FloatType(), True)])


# COMMAND ----------

# DBTITLE 1,Get mode from registry
model_name = dbutils.widgets.get("model")
alias = "production"

model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}@{alias}")

# get return type, ideally the model_type will be information that is stored with the model in the registry 
# which we can retreive here
model_type = "Autoencoder"
return_type = ModelReturnTypes[model_type].value

# COMMAND ----------

# make model selection a widget
def get_yolo_model_files(catalog, schema, cwd) -> None:
    """
    Here we move the ultralytics_yolov5_master folder from volumes to the current working directory as 
    this contains the files needed to run the YOLOv5 model
    """
    source_volume_path = f"dbfs:/Volumes/{catalog}/{schema}/ultralytics_yolov5_master/"
    target_workspace_path = f"file:{cwd}/ultralytics_yolov5_master/"
    dbutils.fs.cp(source_volume_path, target_workspace_path, recurse=True)
    repo_or_dir = f"{cwd}/ultralytics_yolov5_master/"
    
    # Append to path to let workers have access to these files
    sys.path.append(repo_or_dir)

# if model is a YOLO model move in the required model files from the volume
if "yolo" == model_type.lower():
    get_yolo_model_files(catalog, schema, os.getcwd())

# COMMAND ----------

# DBTITLE 1,Model class
class InferenceModel(nn.Module):
    def __init__(self, model, return_type):
        super().__init__()
        self.model = model
        self.model.eval()
        self.return_type = return_type
        if torch.cuda.is_available():
            self.model.cuda()

    def forward(self, input):
        return self.model(input)

# COMMAND ----------

# DBTITLE 1,Test model class
# rows = images.collect()
# content_value = rows[3]["content"] # test with third image in images
# image = Image.open(io.BytesIO(content_value))
# transform = transforms.Compose(
#     [
#         transforms.Resize(128),
#         transforms.ToTensor(),
#     ]
# )

# image = transform(image)
# input = image.unsqueeze(0) # Add batch dimension 
# with torch.no_grad():
#     out = ts_model.model(input)
#     print(out)

# COMMAND ----------

# MAGIC %md
# MAGIC # UDF for inference
# MAGIC

# COMMAND ----------

# DBTITLE 1,Dataset class
class TowerScoutDataset(Dataset):
    """
    Converts image contents into a PyTorch Dataset with preprocessing from nb_model_trainer_development transform_row method.
    """

    def __init__(self, contents):
        self.contents = contents

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        return self._preprocess(self.contents[index])

    def _preprocess(self, content):
        """
        Preprocesses the input image content

        See transform_row method in nb_model_trainer_development nb
        """
        image = Image.open(io.BytesIO(content))

        transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
        return transform(image)

# COMMAND ----------

# DBTITLE 1,UDF for distributed inference
def ts_model_udf(model_fn: InferenceModel, batch_size: int, return_type: StructType): 
    
    @torch.no_grad()
    def predict(content_series_iter):
        model = model_fn()
        for content_series in content_series_iter:
            dataset = TowerScoutDataset(list(content_series))
            loader = DataLoader(dataset, batch_size=batch_size)
            for image_batch in loader:
                output = model(image_batch)
                predicted_labels = output.tolist()
                yield pd.DataFrame(predicted_labels)

    return pandas_udf(return_type, PandasUDFType.SCALAR_ITER)(predict)

# COMMAND ----------

ts_model = InferenceModel(model, return_type)
batch_size = int(dbutils.widgets.get("batch_size"))

# instantiate inference udf
inference_udf = ts_model_udf(model_fn=lambda: ts_model,  batch_size=batch_size, return_type=ts_model.return_type)

# COMMAND ----------

# perform distributed inference with pandas udf
predictions = images.withColumn("prediction", inference_udf(col("content")))

# COMMAND ----------

display(predictions)
