# Databricks notebook source
# MAGIC %md
# MAGIC # NOTE
# MAGIC These notebooks aren't designed to play well with more than one person running them. Rather than to account for that, we use these as guiding examples to demonstrate Spark's capabilities. Be mindful of cells that write or delete content.

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as pst
from pyspark.sql.column import Column

import io
from PIL import Image, ImageStat

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

# Images are stored in dbfs for this example, but not a best practice
source_path = f"{vol_location}/test_images"
checkpoint_path = f"{vol_location}/test_checkpoints"
schema_checkpoint = f"{vol_location}/metadata_schema"
sink_table = f"{cat_name}.{sch_name}.image_metadata"

# Reset Checkpoint
spark.conf.set("cloudFiles.schemaEvolutionMode", "overwrite")
dbutils.fs.rm(checkpoint_path, recurse=True)
dbutils.fs.rm(schema_checkpoint, recurse=True)

# COMMAND ----------

# Reset Table
spark.sql(f"DROP TABLE IF EXISTS {sink_table}")

# COMMAND ----------

# Read Images
image_config = {
    "cloudFiles.format": "binaryFile",
    "pathGlobFilter": "*.jpg"
}

image_df = (
    spark
    .readStream
    .format("cloudFiles")
    .options(**image_config)
    .load(source_path)
)

# Read MetaData
metadata_config = {
    "cloudFiles.format": "json",
    "pathGlobFilter": "*.meta.txt",
    "cloudFiles.schemaLocation": schema_checkpoint,
    "cloudFiles.schemaEvolutionMode": "none",
    "cloudFiles.inferColumnTypes": "true",
}

metadata_df = (
    spark
    .readStream
    .format("cloudFiles")
    .options(**metadata_config)
    .load(source_path)
)

# COMMAND ----------

def extract_file_name(path: "ColumnOrName") -> Column:
    file_with_extension = F.element_at(F.split(path, "/"), -1)
    file_name = F.element_at(F.split(file_with_extension, r"\."), 1)
    return file_name

# COMMAND ----------

# Get file name for join, path is already set for images
image_df = (
    image_df
    .withColumn("file_name", extract_file_name("path"))
)

metadata_df = (
    metadata_df
    .withColumn("file_name", extract_file_name(F.input_file_name()))
)

# COMMAND ----------

images_with_metadata_df = (
    image_df.join(metadata_df, on="file_name")
)

# COMMAND ----------

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

# Add statistics needed for training and EDA
images_with_metadata_df = (
    images_with_metadata_df
    .withColumn("statistics", F.expr("image_statistics_udf(content)"))
)

# COMMAND ----------

 # Write data as a Delta Table
 write_stream = (
    images_with_metadata_df
    .writeStream
    .format("delta")
    .outputMode("append")
    .trigger(availableNow=True)
    .option("checkpointLocation", checkpoint_path)
    .table(sink_table)
)

# COMMAND ----------

batch_df = (
    spark
    .read
    .format("delta")
    .table(sink_table)
)

display(batch_df)

# COMMAND ----------


