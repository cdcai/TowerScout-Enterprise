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

!pip install efficientnet-pytorch ultralytics

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

from tsdb.ml.data_processing import get_bronze_images
from tsdb.utils.uc import CatalogInfo
from tsdb.utils.mlflow import MLFlowHelper
from tsdb.ml.infer import ts_model_udf
from tsdb.ml.data_processing import TowerScoutDataset
from tsdb.ml.efficientnet import EN_classifier
from tsdb.ml.detections import YOLOv5_Detector

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
dbutils.widgets.text("source_schema", defaultValue="towerscout")
dbutils.widgets.text("source_table", defaultValue="image_metadata")
dbutils.widgets.text("batch_size", defaultValue="5") # Randomly picked a number
dbutils.widgets.dropdown("mlflow-alias", "production", ["production", "staging"])

# COMMAND ----------

# Purpose: Retrieve images from a specified Delta table using catalog and schema information from Spark configuration and widgets.
debug_mode = False

# Retrieve catalog information from Spark configuration
catalog_info = CatalogInfo.from_spark_config(spark)

# Get schema and source table from widgets
schema = dbutils.widgets.get("source_schema")
source_table = dbutils.widgets.get("source_table")
alias = dbutils.widgets.get("mlflow-alias")

model_registry = MLFlowHelper(
   catalog=catalog_info.name,
   schema="towerscout"
)

# Create a dropdown widget for model selection
model_options = list(model_registry.registered_models.keys())
dbutils.widgets.dropdown("model", "baseline", model_options)

# table stuff
table_name = f"{catalog_info.name}.{schema}.{source_table}"
cols = ["content", "path"]
images = get_bronze_images(table_name, cols)

if debug_mode:
   display(images.limit(2))

# COMMAND ----------

# DBTITLE 1,Get mode from registry
# Load model from model registry
# The model is expected to adhere to the InferenceModelType protocol
model_name = model_registry.registered_models["baseline"].name
alias = "aws"
ts_model = YOLOv5_Detector.from_uc_registry(model_name=model_name, alias=alias)

# COMMAND ----------

# MAGIC %md
# MAGIC # UDF for inference
# MAGIC

# COMMAND ----------

df = images.toPandas()

# COMMAND ----------

# Create dataset object to apply transformations
dataset: Dataset = TowerScoutDataset(list(df["content"]))
# Create PyTorch DataLoader
loader = DataLoader(dataset, batch_size=4)
for image_batch in loader:
    print(image_batch.size())
    # Perform inference on batch
    output = ts_model.predict(image_batch)
    print(output)
    break

# COMMAND ----------

# Purpose: Retrieve batch size from Databricks widget and instantiate the inference UDF for distributed inference.
# The UDF is created using the ts_model_udf function, which performs batch inference on a Spark DataFrame column.

# retrieve batch size
batch_size = int(dbutils.widgets.get("batch_size"))

yolo_return_type = ArrayType(
    StructType([
        StructField("x1", FloatType(), True),
        StructField("y1", FloatType(), True),
        StructField("x2", FloatType(), True),
        StructField("y2", FloatType(), True),
        StructField("conf", FloatType(), True),
        StructField("class", IntegerType(), True),
        StructField("class_name", StringType(), True),
        StructField("secondary", BooleanType(), True)
    ])
)

# instantiate inference udf
inference_udf = ts_model_udf(model_fn=lambda: ts_model, batch_size=batch_size, return_type=yolo_return_type)

# COMMAND ----------

# Purpose: Perform distributed inference on the 'content' column of the 'images' DataFrame using the instantiated UDF.
# The UDF is applied to each row in the 'content' column to generate predictions.

# perform distributed inference with pandas udf
predictions = images.withColumn("prediction", inference_udf(col("content")))

# COMMAND ----------

# Purpose: Display the predictions DataFrame with the generated predictions.
# The display function is used to render the DataFrame in a rich format.

display(predictions)
