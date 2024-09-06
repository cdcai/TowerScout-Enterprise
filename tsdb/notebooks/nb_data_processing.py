# Databricks notebook source
# MAGIC %run ./utils

# COMMAND ----------

import io
from functools import partial

import numpy as np

from petastorm import TransformSpec
from petastorm.spark.spark_dataset_converter import SparkDatasetConverter

import torchvision

from PIL import Image
from pyspark.sql import DataFrame

# COMMAND ----------

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

def split_data(images: DataFrame) -> (DataFrame, DataFrame, DataFrame):
    """
    Splits a Spark dataframe into train, test, and validation sets.

    Args:
        df (DataFrame): Input dataframe to be split.

    Returns:
        tuple: A tuple containing the train, test, and validation dataframes.
    """
    # split the dataframe into 3 sets
    images_train = images.sampleBy(("label"), fractions={0: 0.8, 1: 0.8})
    images_remaining = images.join(
        images_train, on="path", how="leftanti"
    )  # remaining from images
    images_val = images_remaining.sampleBy(
        ("label"), fractions={0: 0.5, 1: 0.5}
    )  # 50% of images_remaining
    images_test = images_remaining.join(
        images_val, on="path", how="leftanti"
    )  # remaining 50% from the images_remaining
    
    return images_train, images_test, images_val


def split_datanolabel(images: DataFrame) -> (DataFrame, DataFrame, DataFrame):
    """
    Splits a Spark dataframe into train, test, and validation sets.

    Args:
        df (DataFrame): Input dataframe to be split.

    Returns:
        tuple: A tuple containing the train, test, and validation dataframes.
    """
    # split the dataframe into 3 sets
    images_train = images.sample(fraction=0.8)
    images_remaining = images.join(
        images_train, on="path", how="leftanti"
    )  # remaining from images
    images_val = images_remaining.sample(fraction=0.5)  # 50% of images_remaining
    images_test = images_remaining.join(
        images_val, on="path", how="leftanti"
    )  # remaining 50% from the images_remaining
    return images_train, images_test, images_val

# COMMAND ----------

def get_converter_df(dataframe: DataFrame) -> callable:
    """
    Creates a petastrom converter for a Spark dataframe

    Args:
        dataframe: The Spark dataframe
    Returns:
        callable: A petastorm converter 
    """
    
    dataframe = dataframe.transform(compute_bytes, "content")
    converter = create_converter(
        dataframe,
        "bytes"
    )
    
    return converter
