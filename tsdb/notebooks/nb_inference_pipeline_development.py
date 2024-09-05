# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook Overview
# MAGIC
# MAGIC ## Purpose
# MAGIC This notebook is designed to perform inference on image data using a pre-trained model. The process involves reading image metadata from a specified table, processing the images in batches, and applying the model to generate predictions.
# MAGIC
# MAGIC ## Widgets
# MAGIC - **source_schema**: The schema from which to read the image metadata (default: `towerscout_test_schema`).
# MAGIC - **source_table**: The table containing the image metadata (default: `image_metadata`).
# MAGIC - **batch_size**: The number of examples to perform inference on per batch (default: `5`).
# MAGIC - **mlflow_alias**: This will be used to select the appropriate model (default: production)
# MAGIC - **model**: This is the model that will be used for inference pruposes (default: towerscout_model)
# MAGIC
# MAGIC ## Inputs
# MAGIC - Image metadata from the specified schema and table.
# MAGIC
# MAGIC ## Processes
# MAGIC 1. Read image metadata from the specified table.
# MAGIC 2. Load and preprocess images in batches.
# MAGIC 3. Apply a pre-trained model to perform inference on the images.
# MAGIC 4. Store or display the inference results.
# MAGIC
# MAGIC ## Outputs
# MAGIC - Inference results for the processed images.

# COMMAND ----------

# MLflow for model management
import mlflow

# IO operations
import io

# Type hinting
from typing import Protocol, Any, Iterable, Generator

# PyTorch for deep learning
import torch
from torch import nn, Tensor

# Data manipulation
import pandas as pd

# Spark SQL functions and types
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql.types import StructType
from pyspark.sql import DataFrame

# Data classes for structured data
from dataclasses import dataclass

# PyTorch utilities for data handling
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Image processing
from PIL import Image

# OS and system operations
import os
import sys

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %run ./nb_models

# COMMAND ----------

# Purpose: Check if the global view 'global_temp_towerscout_configs' exists and extract configuration values from it. 
# If the view does not exist, exit the notebook with an error message.

# Check if the global view exists
if spark.catalog._jcatalog.tableExists("global_temp.global_temp_towerscout_configs"):
    # Query the global temporary view and collect the first row
    result = spark.sql("SELECT * FROM global_temp.global_temp_towerscout_configs").collect()[0]
    
    # Extract values from the result row
    env = result['env']
    catalog = result['catalog_name']
    schema = result['schema_name']
    debug_mode = result['debug_mode'] == "true"
    unit_test_mode = result['unit_test_mode'] == "true"
else:
    # Exit the notebook with an error message if the global view does not exist
    dbutils.notebook.exit("Global view 'global_temp_towerscout_configs' does not exist, make sure to run the utils notebook")

# COMMAND ----------

# set schema and table to read from, set batch size for number of examples to perform inference on per batch
dbutils.widgets.text("source_schema", defaultValue="towerscout_test_schema")
dbutils.widgets.text("source_table", defaultValue="image_metadata")
dbutils.widgets.text("batch_size", defaultValue="5")

# COMMAND ----------

# set registry to be UC model registry
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Purpose: Define a function to retrieve images from a specified Delta table and return them as a Spark DataFrame.

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
    # Read the Delta table and select the specified columns
    images = spark.read.format("delta").table(table_name).select(columns)
    return images

# COMMAND ----------

# Purpose: Retrieve images from a specified Delta table using catalog and schema information from Spark configuration and widgets.

# Retrieve catalog information from Spark configuration
catalog_info = CatalogInfo.from_spark_config(spark)  # CatalogInfo class defined in utils nb

# Extract catalog name
#catalog = catalog_info.name

# Get schema and source table from widgets
#schema = dbutils.widgets.get("source_schema")
source_table = dbutils.widgets.get("source_table")

# Construct the full table name
table_name = f"{catalog}.{schema}.{source_table}"

# Define the columns to select
cols = ["content", "path"]

# Retrieve images from the specified Delta table
images = get_bronze_images(table_name, cols)

if debug_mode:
   display(images.limit(2))

# COMMAND ----------

# Purpose: Retrieve model names from the Unity Catalog based on the specified catalog and schema, 
#          and create dropdown widgets for model and alias selection.

def get_model_names(catalog: str, schema: str) -> list[str]:
    """
    Retrieve a list of model names from the Unity Catalog based on the specified catalog and schema.

    Parameters:
    catalog (str): The catalog name to filter models.
    schema (str): The schema name to filter models.

    Returns:
    list[str]: A list of model names for models registered in the specified catalog and schema.
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
dbutils.widgets.dropdown("model", "towerscout_model", ts_models)

# Set widget for selecting alias that will be used to select model
aliases = ["production", "staging"]
dbutils.widgets.dropdown("mlflow-alias", "production", aliases)

# COMMAND ----------

# DBTITLE 1,Get mode from registry
# Purpose: Load a model from the model registry based on user selection and prepare it for distributed inference.

# Construct model name and path based on user selection
model_name = f"{catalog}.{schema}.{dbutils.widgets.get('model')}"

# Get alias to look up model by
alias = dbutils.widgets.get('mlflow-alias')

# Load model from model registry
# The model is expected to adhere to the InferenceModelType protocol
ts_model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}@{alias}")

# Define the return type for the inference UDF
return_type = StructType([StructField("logits", ArrayType(ArrayType(FloatType())), True)])

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
# Purpose: Define a pandas UDF for distributed inference with a PyTorch model.
# This function creates a UDF that can be used to perform batch inference on a Spark DataFrame column.

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
    def predict(content_series_iter: Iterable[Any]):
        """
        Predict function to be used within the pandas UDF.

        Args:
            content_series_iter: Iterator over content series.

        Yields:
            DataFrame: DataFrame with predicted labels.
        """
        model = model_fn()  # Load the model
        for content_series in content_series_iter:
            # Create dataset object to apply transformations
            dataset: Dataset = TowerScoutDataset(list(content_series))
            # Create PyTorch DataLoader
            loader = DataLoader(dataset, batch_size=batch_size)
            for image_batch in loader:
                # Perform inference on batch
                output = model(image_batch)
                predicted_labels = output.tolist()
                yield pd.DataFrame(predicted_labels)

    return pandas_udf(return_type, PandasUDFType.SCALAR_ITER)(predict)

# COMMAND ----------

# Purpose: Retrieve batch size from Databricks widget and instantiate the inference UDF for distributed inference.
# The UDF is created using the ts_model_udf function, which performs batch inference on a Spark DataFrame column.

# retrieve batch size
batch_size = int(dbutils.widgets.get("batch_size"))

# instantiate inference udf
inference_udf = ts_model_udf(model_fn=lambda: ts_model, batch_size=batch_size, return_type=return_type)

# COMMAND ----------

# Purpose: Perform distributed inference on the 'content' column of the 'images' DataFrame using the instantiated UDF.
# The UDF is applied to each row in the 'content' column to generate predictions.

# perform distributed inference with pandas udf
predictions = images.withColumn("prediction", inference_udf(col("content")))

# COMMAND ----------

# Purpose: Display the predictions DataFrame with the generated predictions.
# The display function is used to render the DataFrame in a rich format.

display(predictions)
