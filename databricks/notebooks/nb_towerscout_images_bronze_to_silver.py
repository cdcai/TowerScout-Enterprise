# Databricks notebook source
import pyspark.sql.functions as F
import pyspark.sql.types as pst
from pyspark.sql.column import Column

import io
from PIL import Image, ImageStat

# COMMAND ----------

# MAGIC %md
# MAGIC # Bronze to Silver Pipeline
# MAGIC Using an arbitrary image data source, this notebook uses AutoLoader to read images as binary. AutoLoader is only available for streams, so this notebook utilizes TriggerOnce to simulate batch behavior but still getting the benefits of CloudFiles.
# MAGIC
# MAGIC Extensions of this notebook could include the extraction of EXIF metadata or other metadata that tends to be embedded in jpg and png file formats.
# MAGIC
# MAGIC ## Requirements
# MAGIC To run this notebook, you need a volume and schema defined in Unity Catalog. EDAV sets catalog name information in the cluster.

# COMMAND ----------

# Fetch the value of the parameter spark.databricks.sql.initial.catalog.name from Spark config, exit if the value is not set in cluster configuration
initial_catalog_name = spark.conf.get("spark.databricks.sql.initial.catalog.name")

if not initial_catalog_name:
    dbutils.notebook.exit("Initial catalog name is empty in cluster")
else:
    cat_name = initial_catalog_name
    display(cat_name)
   
    # Fetch and display info table
    schema_info = spark.sql(
        f"SELECT volume_schema, storage_location FROM {cat_name}.information_schema.volumes"
    ).collect()
   
    if not schema_info:
        dbutils.notebook.exit("No schema exists in the catalog")
    else:
        sch_name = schema_info[0]["volume_schema"]
        display(sch_name)
 
    vol_location = schema_info[0]["storage_location"]
    if vol_location:
        display(vol_location)
 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters
# MAGIC Since we extract volume and schema information from the cluster, we only need directory or table names

# COMMAND ----------

dbutils.widgets.text("source_volume_directory", defaultValue="test_images")
dbutils.widgets.text("checkpoint_directory", defaultValue="test_checkpoints")
dbutils.widgets.text("sink_table_name", defaultValue="test_image_silver")

# COMMAND ----------

# Get parameters and set required notebook variables
parameters = dbutils.widgets.getAll()

source_path = f"{vol_location}/{parameters['source_volume_directory']}"
checkpoint_path = f"{vol_location}/checkpoints/{parameters['checkpoint_directory']}"
sink_table = f"{cat_name}.{sch_name}.{parameters['sink_table_name']}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Image Statistics
# MAGIC Image preprocessing for deep learning often leverages image statistics. We precompute these and store them. 

# COMMAND ----------

#TODO: This goes into a code file
statistics_schema = pst.StructType([
    pst.StructField("mean", pst.ArrayType(pst.DoubleType())),
    pst.StructField("median", pst.ArrayType(pst.IntegerType())),
    pst.StructField("stddev", pst.ArrayType(pst.DoubleType())),
    pst.StructField("extrema", pst.ArrayType(pst.ArrayType(pst.IntegerType()))),
])


def image_statistics_udf(image_binary: pst.BinaryType) -> statistics_schema:
    image = Image.open(io.BytesIO(image_binary))
    image_statistics = ImageStat.Stat(image)

    return {
        "mean": image_statistics.mean,
        "median": image_statistics.median,
        "stddev": image_statistics.stddev,
        "extrema": image_statistics.extrema,
    }

_ = spark.udf.register("image_statistics_udf", image_statistics_udf, statistics_schema)


# COMMAND ----------

 # Read Images
image_config = {
    "cloudFiles.format": "binaryFile",
    "pathGlobFilter": "*.jpg"
}

source_df = (
    spark
    .readStream
    .format("cloudFiles")
    .options(**image_config)
    .load(source_path)
)


# Add statistics needed for training and EDA
transformed_df = (
    source_df
    .withColumn("statistics", F.expr("image_statistics_udf(content)"))
)
 
 # Write data as a Delta Table
 write_stream = (
    transformed_df
    .writeStream
    .format("delta")
    .outputMode("append")
    .trigger(availableNow=True)
    .option("checkpointLocation", checkpoint_path)
    .table(sink_table)
)
