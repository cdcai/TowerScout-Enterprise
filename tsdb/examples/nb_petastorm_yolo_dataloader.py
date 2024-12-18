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

# # Fetch the value of the parameter spark.databricks.sql.initial.catalog.name from Spark config, exit if the value is not set in cluster configuration
# initial_catalog_name = spark.conf.get("spark.databricks.sql.initial.catalog.name")
# if not initial_catalog_name:
#     dbutils.notebook.exit("Initial catalog name is empty in cluster")
# else:
#     cat_name = initial_catalog_name
#     display(cat_name)
   
#     # Fetch and display info table
#     schema_info = spark.sql(
#         f"SELECT volume_schema, storage_location FROM {cat_name}.information_schema.volumes"
#     ).collect()
   
#     if not schema_info:
#         dbutils.notebook.exit("No schema exists in the catalog")
#     else:
#         sch_name = schema_info[0]["volume_schema"]
#         display(sch_name)
 
#     vol_location = schema_info[0]["storage_location"]
#     if vol_location:
#         display(vol_location)
 

# COMMAND ----------

petastorm_path = "file:///dbfs/tmp/petastorm/cache"
catalog = "edav_dev_csels"
schema = "towerscout" # hardcode these for now

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.column import Column

# images = (
#     spark
#     .table(f"{catalog}.{schema}.test_image_silver")
#     .select("image_path", "bboxes")
# )


images = spark.sql(
    f"""
    SELECT image_path, bboxes FROM {catalog}.{schema}.test_image_silver
    WHERE request_id in ("019d0e1b", "026d23e0", "048cd78f")
    """
)

def get_proper_path(path: "ColumnOrName") -> Column:
    # remove the dbfs: prefix as that causes errors when reading
    file_with_extension = F.element_at(F.split(path, ":"), 2)
    return file_with_extension

images = (
    images
    .withColumn("image_path", get_proper_path("image_path"))
)


# COMMAND ----------

display(images)

# COMMAND ----------

# MAGIC %md
# MAGIC 6*4 + 4 + 2 + 36
# MAGIC
# MAGIC floats + int + class name "ct" 2 chars long + size of keys (num of chars in key strings)

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
        F.lit(4) + F.length("image_path") + (6*4 + 4 + 2 + 36)*F.size("bboxes") 
    )
    .groupBy()
    .agg(F.sum("bytes").alias("bytes"))
    .collect()[0]["bytes"]
)

# Cache
converter = make_spark_converter(
    images, 
    parquet_row_group_size_bytes=int(num_bytes/sc.defaultParallelism)
)

# COMMAND ----------

display(images)

# COMMAND ----------

def transform_row(batch_pd):
    """
    Defines how to transform partition elements
    """
    transformers = [
        #torchvision.transforms.Lambda(lambda image: Image.open(io.BytesIO(image)))
        torchvision.transforms.Lambda(lambda image: Image.open(image))
    ]

    transformers.extend([
        torchvision.transforms.Resize(320),
        torchvision.transforms.ToTensor(),
    ])

    transformer_pipeline = torchvision.transforms.Compose(transformers)

    box_transform = [
        torchvision.transforms.Lambda(lambda boxes: torch.tensor([[box["x1"], box["y1"], box["x2"], box["y2"]] for box in boxes]))
    ]

    #batch_pd["im_file"] = batch_pd["image_path"].map(lambda img_path: img_path)
    print(batch_pd.columns.tolist())
    batch_pd["img"] = batch_pd["image_path"].map(lambda image_path: np.ascontiguousarray(transformer_pipeline(image_path).numpy()))
    batch_pd["bboxes_trans"] = batch_pd["bboxes"].map(lambda boxes: np.ascontiguousarray(box_transform(boxes).numpy()))

    return batch_pd[["img"]], batch_pd[["bboxes_trans"]]

def get_transform_spec():
    """
    Applies transforms across partitions
    """
    spec = TransformSpec(
        partial(transform_row),
        edit_fields=[
            ("img", np.float32, (3, 320, 320), False),
            ("bboxes_trans", np.float32, (-1, 4), False)
        ],
        selected_fields=["img", "bboxes_trans"], 
    )

    return spec

# COMMAND ----------

context_args = {
    "transform_spec": get_transform_spec(),
    "batch_size": 2
}

with converter.make_torch_dataloader(**context_args) as dataloader:
    for image in dataloader:
        print(f"\n\nshape of features key: {image['features'].shape}\n\n")
        #plt.imshow(image["features"].squeeze(0).permute(1, 2, 0))
        break

# COMMAND ----------

# Clear Petastorm Cache
converter.delete()

