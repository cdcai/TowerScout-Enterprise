# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook Overview
# MAGIC
# MAGIC ## Purpose
# MAGIC This notebook is designed to perform inference on image data using a pre-trained model. The process involves reading image metadata from a specified table, processing the images in batches, and applying the model to generate predictions.
# MAGIC
# MAGIC ## Inputs
# MAGIC - Images with exif metadata from the TowerScout front end
# MAGIC
# MAGIC ## Processes
# MAGIC 1. Read images as a binary and metadata from the specified table.
# MAGIC 2. Load and preprocess images in batches.
# MAGIC 3. Apply a pre-trained model to perform inference on the images.
# MAGIC 4. Store or display the inference results.
# MAGIC
# MAGIC ## Outputs
# MAGIC - Inference results for the processed images.

# COMMAND ----------

# MAGIC %run ./nb_config_retrieval

# COMMAND ----------

from tsdb.utils.uc import CatalogInfo
import tsdb.preprocessing.transformations as trf
from tsdb.preprocessing.images import make_image_metadata_udf
from tsdb.ml.infer import make_towerscout_predict_udf

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
    bronze_path = result['bronze_path']
    silver_table_name = result['silver_table_name']
    writestream_trigger_args = result['writestream_trigger_args'].asDict()
    image_config = result['image_config'].asDict()
    batch_size = int(result['batch_size'])
    checkpoint_path = result['checkpoint_path']
else:
    # Exit the notebook with an error message if the global view does not exist
    dbutils.notebook.exit("Global view 'global_temp_towerscout_configs' does not exist, make sure to run the utils notebook")

# COMMAND ----------

# table stuff
image_directory_path = f"{bronze_path}/*/*"
sink_table = f"{catalog}.{schema}.{silver_table_name}"

# Create our UDFs
# Batch size is a very important parameter, since we iterate through images to process them
towerscout_inference_udf = make_towerscout_predict_udf(catalog, schema, batch_size=100)
image_metadata_udf = make_image_metadata_udf(spark)

# COMMAND ----------

# Read Images
image_df = (
    spark
    .readStream
    .format("cloudFiles")
    .options(**image_config)
    .option("clouldFiles.useNotifications", "true")
    .load(image_directory_path) # parameterize
    .repartition(8)
)

transformed_df = (
    image_df
    .transform(trf.parse_file_path)
    .transform(trf.perform_inference, towerscout_inference_udf)
    .transform(trf.extract_metadata, image_metadata_udf)
    .transform(trf.current_time)
    .transform(trf.hash_image)
    .selectExpr(
        "user_id",
        "request_id",
        "uuid",
        "processing_time",
        "results.bboxes as bboxes",
        "image_hash",
        "path as image_path",
        "image_id",
        "results.model_version as model_version",
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

        # Option 2 (Default option for streams checks every 500ms for new files)
        # Have stream run continuously between two times
        # trigger = Continuous (processingTime="5 minutes") "10 seconds", "5 milliseconds"

        # Option 3
        # Trigger workflow on file arrival, but it only works on directories with less than 10,000 files
        .trigger(availableNow=True) 
        .option("checkpointLocation", checkpoint_path) # parameterized
        .table(sink_table)
    )

# COMMAND ----------


