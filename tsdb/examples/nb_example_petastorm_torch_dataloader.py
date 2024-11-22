# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # NOTE
# MAGIC These notebooks aren't designed to play well with more than one person running them. Rather than to account for that, we use these as guiding examples to demonstrate Spark's capabilities. Be mindful of cells that write or delete content.

# COMMAND ----------

import io
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import pyspark.sql.functions as F

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec

import torchvision
import torch

from PIL import Image

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

petastorm_path = "file:///dbfs/tmp/petastorm/cache"

# COMMAND ----------

images = (
    spark
    .table(f"{cat_name}.{sch_name}.image_metadata")
    .select("content", "path")
)

# COMMAND ----------

# Create petastorm cache
spark.conf.set(
    SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, 
    petastorm_path
)

# Calculate bytes
num_bytes = (
    images
    .withColumn(
        "bytes", 
        F.lit(4) + F.length("content")).groupBy().agg(F.sum("bytes").alias("bytes")).collect()[0]["bytes"]
)

# Cache
converter = make_spark_converter(
    images, 
    parquet_row_group_size_bytes=int(num_bytes/sc.defaultParallelism)
)

# COMMAND ----------

def transform_row(batch_pd):
    """
    Defines how to transform partition elements
    """
    transformers = [
        torchvision.transforms.Lambda(lambda image: Image.open(io.BytesIO(image)))
    ]

    transformers.extend([
        torchvision.transforms.Resize(128),
        torchvision.transforms.ToTensor(),
    ])

    transformer_pipeline = torchvision.transforms.Compose(transformers)

    # Needs to be row-major array
    batch_pd["features"] = (
        batch_pd["content"]
        .map(
            lambda image: np.ascontiguousarray(transformer_pipeline(image).numpy())
        )
    )

    return batch_pd[["features"]]

def get_transform_spec():
    """
    Applies transforms across partitions
    """
    spec = TransformSpec(
        partial(transform_row),
        edit_fields=[
            ("features", np.float32, (3, 128, 128), False),
        ],
        selected_fields=["features"]
    )

    return spec

# COMMAND ----------

context_args = {
    "transform_spec": get_transform_spec(),
    "batch_size": 1
}

with converter.make_torch_dataloader(**context_args) as dataloader:
    for image in dataloader:
        plt.imshow(image["features"].squeeze(0).permute(1, 2, 0))
        break

# COMMAND ----------

# Clear Petastorm Cache
converter.delete()

