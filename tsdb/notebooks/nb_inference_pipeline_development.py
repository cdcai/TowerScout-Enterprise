# Databricks notebook source
import mlflow

from typing import Any, Protocol

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

# MAGIC %run ./nb_models

# COMMAND ----------

# MAGIC %run ./dataloader_development

# COMMAND ----------

# set schema and table to read from, set batch size for number of examples to perform inference on per batch
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
        ts_models.append(model.name.split(".")[-1])

dbutils.widgets.dropdown("model", ts_models[0], ts_models)

# set widget for selecting alias that will be used to select model
aliases = ["production", "staging"]
dbutils.widgets.dropdown("mlflow-alias", "production", aliases)

# COMMAND ----------

# DBTITLE 1,Get mode from registry
# construct model name and path based on user selection
model_name = f"{catalog}.{schema}.{dbutils.widgets.get('model')}"

# get alias to look up model by
alias = dbutils.widgets.get('mlflow-alias')

# load model from model registry, the assumption is this will have methods outlined in the InferenceModelType protocol
# in the nb_models notebook so should have attributes like return_type and methods like get_model_files that are called upon instantiation as well as a method called predict
ts_model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}@{alias}")

return_type = ts_model.model_type

# COMMAND ----------

# MAGIC %md
# MAGIC # UDF for inference
# MAGIC

# COMMAND ----------

# DBTITLE 1,Dataset class
class TowerScoutDataset(Dataset):
    """
    Converts image contents into a PyTorch Dataset with preprocessing 
    from the nb_model_trainer_development transform_row() method.
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

        See transform_row() method in nb_model_trainer_development nb
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
def ts_model_udf(model_fn: InferenceModelType, batch_size: int, return_type: StructType) -> DataFrame: 
    """
    A pandas UDF for distributed inference with a PyTorch model
    """
    @torch.no_grad()
    def predict(content_series_iter):
        model = model_fn() 
        for content_series in content_series_iter:
            dataset: Dataset = TowerScoutDataset(list(content_series)) # create dataset object to apply transformations
            loader = DataLoader(dataset, batch_size=batch_size) # create PyTorch dataloader
            for image_batch in loader: # iterate through dataloader
                output = model.predict(image_batch) # perform inference on batch
                predicted_labels = output.tolist()
                yield pd.DataFrame(predicted_labels)

    return pandas_udf(return_type, PandasUDFType.SCALAR_ITER)(predict)

# COMMAND ----------

# retrieve batch size
batch_size = int(dbutils.widgets.get("batch_size"))

# instantiate inference udf
inference_udf = ts_model_udf(model_fn=lambda: ts_model,  batch_size=batch_size, return_type=ts_model.return_type)

# COMMAND ----------

# perform distributed inference with pandas udf
predictions = images.withColumn("prediction", inference_udf(col("content")))

# COMMAND ----------

display(predictions)
