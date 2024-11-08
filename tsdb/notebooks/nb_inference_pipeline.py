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

!pip install efficientnet_pytorch
!pip install opencv-python # need for yolo model
!pip install ultralytics==8.2.92 # need for yolo model
!pip install gitpython==3.1.30 pillow==10.3.0 requests==2.32.0 setuptools==70.0.0 # need for loading yolo with torch.hub.load from ultralytics
!pip install shapely==2.0.3
!pip install mlflow-skinny==2.15.1 # need this if on GPU 
dbutils.library.restartPython() # need this if on GPU

# COMMAND ----------

# MLflow for model management
import mlflow

# IO operations
import io

# PyTorch for deep learning
import torch

# Data manipulation
import pandas as pd

# Spark SQL functions and types
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType, IntegerType, StringType, BooleanType

# PyTorch utilities for data handling
from torch.utils.data import Dataset, DataLoader

# OS and system operations
import os
import sys

from tsdb.utils.uc import CatalogInfo
from tsdb.utils.mlflow import MLFlowHelper
from tsdb.ml.infer import ts_model_udf
from tsdb.ml.data_processing import TowerScoutDataset
from tsdb.ml.efficientnet import ENClassifier
from tsdb.ml.detections import YOLOv5_Detector, initialize_yolov5_cluster
from tsdb.utils.uc import CatalogInfo

# COMMAND ----------

# # Purpose: Check if the global view 'global_temp_towerscout_configs' exists and extract configuration values from it. 
# # If the view does not exist, exit the notebook with an error message.

# # Check if the global view exists
# if spark.catalog._jcatalog.tableExists("global_temp.global_temp_towerscout_configs"):
#     # Query the global temporary view and collect the first row
#     result = spark.sql("SELECT * FROM global_temp.global_temp_towerscout_configs").collect()[0]
    
#     # Extract values from the result row
#     env = result['env']
#     catalog = result['catalog_name']
#     schema = result['schema_name']
#     debug_mode = result['debug_mode'] == "true"
#     unit_test_mode = result['unit_test_mode'] == "true"
# else:
#     # Exit the notebook with an error message if the global view does not exist
#     dbutils.notebook.exit("Global view 'global_temp_towerscout_configs' does not exist, make sure to run the utils notebook")

# COMMAND ----------

# set schema and table to read from, set batch size for number of examples to perform inference on per batch
dbutils.widgets.text("source_schema", defaultValue="towerscout")
dbutils.widgets.text("source_table", defaultValue="image_metadata")
dbutils.widgets.text("batch_size", defaultValue="5") # Randomly picked a number
dbutils.widgets.dropdown("mlflow-alias", "production", ["production", "staging"])

# COMMAND ----------

def get_bronze_images(
    table_name: str, columns: list[str]
):
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
debug_mode = False

# Retrieve catalog information from Spark configuration
catalog_info = CatalogInfo.from_spark_config(spark)

# Get schema and source table from widgets
catalog = catalog_info.name
schema = dbutils.widgets.get("source_schema")
source_table = dbutils.widgets.get("source_table")
alias = dbutils.widgets.get("mlflow-alias")

model_registry = MLFlowHelper(
   catalog=catalog_info.name,
   schema="towerscout"
)

# Create a dropdown widget for model selection
model_options = list(model_registry.registered_models.keys())
# dbutils.widgets.dropdown("model", "baseline", model_options)

# table stuff
table_name = f"{catalog_info.name}.{schema}.{source_table}"
cols = ["content", "path"]
images = get_bronze_images(table_name, cols)

if debug_mode:
   display(images.limit(2))

# COMMAND ----------

from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from typing import Any, Iterable

from torch.utils.data import DataLoader

import pandas as pd

from tsdb.ml.models import InferenceModelType
from tsdb.ml.data_processing import TowerScoutDataset

# COMMAND ----------

from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from typing import Any, Iterable

from torch.utils.data import DataLoader

import pandas as pd

from tsdb.ml.models import InferenceModelType
from tsdb.ml.data_processing import TowerScoutDataset

# COMMAND ----------

import sys

yolo_version = "v5"
path = f"/Volumes/{catalog}/{schema}/misc/yolo{yolo_version}"
sc.addPyFile(path)

if path not in sys.path:
    sys.path.append(path)

# COMMAND ----------

def make_towerscout_predict_udf(
    catalog: str,
    schema: str,
    # yolo_alias: str,
    batch_size: int,
) -> DataFrame:
    """
    For the predict_batch_udf, we need the outer function to initialize the models
    and the inner function to perform the inference.

    Args:
        model_fn (InferenceModelType): The PyTorch model.
        batch_size (int): Batch size for the DataLoader.
        return_type (StructType): Return type for the UDF.

    Returns:
        DataFrame: DataFrame with predictions.
    """    
    yolo_model_name = f"{catalog}.{schema}.yolo_autoshape" 
    en_model_name = f"{catalog}.{schema}.efficientnet"  
    alias = "aws"

    # Retrieves models by alias and create inference objects
    yolo_detector = YOLOv5_Detector.from_uc_registry(
        model_name=yolo_model_name,
        alias=alias,
        batch_size=batch_size,
    )

    # We nearly always use efficientnet for classification but you don't have to
    en_classifier = ENClassifier.from_uc_registry(
        model_name=en_model_name,
        alias=alias
    )

    @torch.no_grad()
    def predict(content_series_iter: Iterable[Any]):
        """
        Predict function to be used within the pandas UDF.

        Args:
            content_series_iter: Iterator over content series.

        Yields:
            DataFrame: DataFrame with predicted labels.
        """
        for content_series in content_series_iter:
            # Create dataset object to apply transformations
            dataset: Dataset = TowerScoutDataset(list(content_series))
            # Create PyTorch DataLoader
            loader = DataLoader(dataset, batch_size=batch_size)
            for image_batch in loader:
                # Perform inference on batch
                output = yolo_detector.predict(model_input=image_batch, secondary=en_classifier)
                predicted_labels = output.tolist()
                yield pd.Series(predicted_labels)

    return pandas_udf(yolo_detector.return_type, PandasUDFType.SCALAR_ITER)(predict)

# COMMAND ----------

# MAGIC %md
# MAGIC # UDF for inference
# MAGIC

# COMMAND ----------

towerscout_inference_udf = make_towerscout_predict_udf(catalog, schema, 100)

# COMMAND ----------

# Purpose: Perform distributed inference on the 'content' column of the 'images' DataFrame using the instantiated UDF.
# The UDF is applied to each row in the 'content' column to generate predictions.

# perform distributed inference with pandas udf
predictions = images.withColumn("prediction", towerscout_inference_udf(col("content")))

# COMMAND ----------

# Purpose: Display the predictions DataFrame with the generated predictions.
# The display function is used to render the DataFrame in a rich format.

display(predictions)

# COMMAND ----------


