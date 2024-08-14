# Databricks notebook source
# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %run ./demo_model

# COMMAND ----------

import io
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from petastorm import TransformSpec

import torchvision

from PIL import Image

# COMMAND ----------

dbutils.widgets.text("source_table", defaultValue="image_metadata")

# COMMAND ----------

catalog_info = CatalogInfo.from_spark_config(spark)
schema = catalog_info.schemas[0]

source_table = dbutils.widgets.get("source_table")
table_name = f"{catalog_info.name}.{schema.name}.{source_table}"

# project name folder
petastorm_path = "file:///dbfs/TowerScout/tmp/petastorm/dataloader_development_cache"

# Create petastorm cache
spark.conf.set(
    SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, 
    petastorm_path
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

images_df = (
    spark
    .table(table_name)
    .select("content", "path")
    .transform(compute_bytes, "content")
)

# Create a Petastorm converter for the images DataFrame
converter = create_converter(images_df, "bytes")

transformers = [
    torchvision.transforms.Lambda(lambda image: Image.open(io.BytesIO(image))),
    torchvision.transforms.Resize(128),
    torchvision.transforms.ToTensor()
]

# Define arguments for the dataloader context
batch_size = 8
context_args = {
    "transform_spec": get_transform_spec(),  
    "batch_size": batch_size
}

trainer = ModelTrainer({"lr": 1e-3})
converter_length = len(converter)
steps_per_epoch = converter_length // batch_size


# COMMAND ----------

# MAGIC %md
# MAGIC # Note
# MAGIC This runs infinitely, need to limit the steps

# COMMAND ----------

with converter.make_torch_dataloader(**context_args) as dataloader:
    dataloader_iter = iter(dataloader)
    
    for epoch in range(1):
        for minibatch_num in range(steps_per_epoch):
            minibatch_images = next(dataloader_iter)
            metrics = trainer.training_step(minibatch_images)
            print(minibatch_num, metrics)

# COMMAND ----------

# Clear Petastorm Cache
converter.delete()


# COMMAND ----------


