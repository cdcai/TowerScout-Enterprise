# Databricks notebook source
# MAGIC %run ./utils

# COMMAND ----------

import io
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from petastorm import TransformSpec

import torchvision
import torch

from PIL import Image

# COMMAND ----------

dbutils.widgets.text("source_table", defaultValue="image_metadata")

# COMMAND ----------

catalog_info = CatalogInfo.from_spark_config(spark)
schema = catalog_info.schemas[1]

source_table = dbutils.widgets.get("source_table")
table_name = f"{catalog_info.name}.{schema.name}.{source_table}"

petastorm_path = "file:///dbfs/tmp/petastorm/cache"

# Create petastorm cache
spark.conf.set(
    SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, 
    petastorm_path
)

# COMMAND ----------

images = (
    spark
    .table(table_name)
    .select("content", "path")
    .transform(compute_bytes, "content")
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

def get_transform_spec(transform_function):
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

converter = create_converter(images, "bytes")

context_args = {
    "transform_spec": get_transform_spec(),
    "batch_size": 1
}

with converter.make_torch_dataloader(**context_args) as dataloader:
    dataloader_iter = iter(dataloader)
    for image in dataloader:
        plt.imshow(image["features"].squeeze(0).permute(1, 2, 0))
        break

# COMMAND ----------

for i, batch in enumerate(dataloader_iter):
    print(i, batch["features"].size())

# COMMAND ----------

# Clear Petastorm Cache
converter.delete()

