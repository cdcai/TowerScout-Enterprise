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

from tsdb.utils.uc import CatalogInfo
import tsdb.preprocessing.transformations as trf
from tsdb.preprocessing.images import make_image_metadata_udf
from tsdb.ml.infer import make_towerscout_predict_udf

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

# Purpose: Retrieve images from a specified Delta table using catalog and schema information from Spark configuration and widgets.
debug_mode = False

# Retrieve catalog information from Spark configuration
catalog_info = CatalogInfo.from_spark_config(spark)

# Get schema and source table from widgets
catalog = catalog_info.name
schema = "towerscout" # delete

# table stuff
image_directory_path = "/Volumes/edav_dev_csels/towerscout/images/maps/bronze/*/*"
sink_table = f"{catalog}.{schema}.test_image_silver"

# Create our UDFs
# Batch size is a very important parameter, since we iterate through images to process them
towerscout_inference_udf = make_towerscout_predict_udf(catalog, schema, batch_size=100)
image_metadata_udf = make_image_metadata_udf(spark)

# COMMAND ----------

import pyspark.sql.functions as F

struct_literal = F.struct(
    F.lit("yolo_autoshape").alias("yolo_model"),
    F.lit("1").alias("yolo_model_version"),
    F.lit("efficientnet").alias("efficientnet_model"),
    F.lit("1").alias("efficientnet_model_version")
)

# COMMAND ----------

# Read Images
image_config = {
    "cloudFiles.format": "binaryFile",
    "pathGlobFilter": "*.jpeg" # */*/*.jpeg
}

image_df = (
    spark
    .readStream
    .format("cloudFiles")
    .options(**image_config)
    .load(image_directory_path) # parameterize
)

transformed_df = (
    image_df
    .transform(trf.parse_file_path)
    .transform(trf.perform_inference, towerscout_inference_udf)
    .transform(trf.extract_metadata, image_metadata_udf)
    .transform(trf.current_time)
    .transform(trf.hash_image)
    .withColumn("model_version", struct_literal)
    .selectExpr(
        "user_id",
        "request_id",
        "uuid",
        "processing_time",
        "bboxes",
        "image_hash",
        "path as image_path",
        "model_version",
        "image_metadata",
        "map_provider"
    )
)

if debug_mode:
    (
        transformed_df
        # limit here if desired
        .writeStream
        .queryName("inference_predictions")
        .format("memory")
        .start()
    )
else:
    write_stream = (
        transformed_df
        .writeStream
        .format("delta")
        .outputMode("append")
        # Parameterize (available now stops the stream once available data has been processed)
        # Option 1: Schedule the workflow to run every n minutes
        # trigger is every n seconds, 5 minutes hard real-time requirement

        # Option 2
        # Have stream run continuously between two times
        # trigger = Continuous (processingTime="5 minutes") "10 seconds", "5 milliseconds"

        # Option 3
        # Trigger workflow on file arrival, but it only works on directories with less than 10,000 files
        .trigger(availableNow=True) 
        .option("checkpointLocation", "file:/tmp/checkpoints_2") # parameterized
        .table(sink_table)
    )

# COMMAND ----------

display(spark.sql("SELECT * from inference_predictions"))

# COMMAND ----------


