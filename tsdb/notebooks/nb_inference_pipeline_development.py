# Databricks notebook source
import mlflow

from typing import Protocol, Any, Iterable, Generator

import torch
from torch import nn, Tensor

import pandas as pd

from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql.types import StructType
from pyspark.sql import DataFrame

from dataclasses import dataclass

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

import os
import sys
import io

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %run ./nb_models

# COMMAND ----------

# set schema and table to read from, set batch size for number of examples to perform inference on per batch
dbutils.widgets.text("source_schema", defaultValue="towerscout_test_schema")
dbutils.widgets.text("source_table", defaultValue="image_metadata")
dbutils.widgets.text("batch_size", defaultValue="5")

# COMMAND ----------

# set registry to be UC model registry
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

def get_bronze_images(
    table_name: str, columns: list[str]
) -> DataFrame:
    """
    Retrieve images from a Delta table.

    Parameters:
    table_name (str): The name of the table to read from.
    columns (list[str]): The list of columns to select.

    Returns:
    DataFrame: A Spark DataFrame containing the selected columns from the table.
    """
    images = spark.read.format("delta").table(table_name).select(columns)
    return images

# COMMAND ----------

# Retrieve catalog information from Spark configuration
catalog_info = CatalogInfo.from_spark_config(
    spark
)  # CatalogInfo class defined in utils nb

# Extract catalog name
catalog = catalog_info.name

# Get schema and source table from widgets
schema = dbutils.widgets.get("source_schema")
source_table = dbutils.widgets.get("source_table")

# Construct the full table name
table_name = f"{catalog}.{schema}.{source_table}"

# Define the columns to select
cols = ["content", "path"]

# Retrieve images from the specified Delta table
images = get_bronze_images(table_name, cols)

# COMMAND ----------

def get_model_names(catalog: str, schema: str) -> list[str]:
    """
    Retrieve a list of model names from the Unity Catalog based on the specified catalog and schema.

    Parameters:
    catalog (str): The catalog name to filter models.
    schema (str): The schema name to filter models.

    Returns:
    list[str]: A list of model names for models regi
    """
    registered_models = mlflow.search_registered_models()

    # Initialize an empty list to store filtered model names
    ts_models = []

    # Loop over registered models to retrieve models in the correct catalog and schema
    for model in registered_models:
        catalog_name, schema_name, name = model.name.split(".")
        if catalog_name == catalog and schema_name == schema:
            ts_models.append(model.name.split(".")[-1])

    return ts_models

# Retrieve model names based on the specified catalog and schema
ts_models = get_model_names(catalog, schema)

# Create a dropdown widget for model selection
dbutils.widgets.dropdown("model", ts_models[0], ts_models)

# Set widget for selecting alias that will be used to select model
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

# get model return type for inference UDF
if hasattr(ts_model, 'return_type'):
    return_type = ts_model.return_type
else:
    raise AttributeError("The loaded model does not have the attribute 'return_type'. Model must adhere to the InferenceModelType protocol. Please load a model which does.")

if not hasattr(model, 'predict'):
    raise AttributeError("The model does not have a 'predict' method")

# COMMAND ----------

# MAGIC %md
# MAGIC # UDF for inference
# MAGIC

# COMMAND ----------

# DBTITLE 1,Dataset class
@dataclass
class TowerScoutDataset(Dataset):
    """
    Converts image contents into a PyTorch Dataset with preprocessing from nb_model_trainer_development transform_row method.
    """
    contents: Iterable[Any] = None

    def __len__(self) -> int:
        return len(self.contents)

    def __getitem__(self, index) -> Tensor:
        return self._preprocess(self.contents[index])

    def _preprocess(self, content) -> Tensor:
        """
        Preprocesses the input image content

        See transform_row method in nb_model_trainer_development nb
        """
        image = Image.open(io.BytesIO(content))
        # maybe make transform a callable argument to func/class
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
    A pandas UDF for distributed inference with a PyTorch model.

    Args:
        model_fn (InferenceModelType): The PyTorch model.
        batch_size (int): Batch size for the DataLoader.
        return_type (StructType): Return type for the UDF.

    Returns:
        DataFrame: DataFrame with predictions.
    """
    @torch.no_grad()
    def predict(content_series_iter: Iterable[Any]) -> Generator[pd.DataFrame, None, None]:
        """
        Predict function to be used within the pandas UDF.

        Args:
            content_series_iter: Iterator over content series.

        Yields:
            DataFrame: DataFrame with predicted labels.
        """
        model = model_fn()
        for content_series in content_series_iter:
            # Create dataset object to apply transformations
            dataset: Dataset = TowerScoutDataset(list(content_series))
            # Create PyTorch DataLoader
            loader = DataLoader(dataset, batch_size=batch_size)
            for image_batch in loader:
                # Perform inference on batch
                output = model.predict(image_batch)
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
