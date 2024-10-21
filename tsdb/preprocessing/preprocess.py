"""
This module contains higher level preprocessing workflows, that use a combination of tsdb.preprocessing.functions
"""
from pyspark.context import SparkContext

from petastorm.spark import SparkDatasetConverter, make_spark_converter

from tsdb.preprocessing.functions import sum_column, sum_bytes

import torch
from torchvision.transforms import v2
from torchvision.transforms import Compose


def create_converter(
    dataframe, bytes_column: "ColumnOrName", sc: SparkContext, parallelism: int = 0
) -> SparkDatasetConverter:
    """
    Returns a PetaStorm converter created from dataframe.

    Args:
        dataframe: DataFrame
        byte_column: Column that contains the byte count. Used to create the petastorm  cache
        parallelism: integer for parallelism, used to create the petastorm cache
    """
    # Note this uses spark context
    if parallelism == 0:
        parallelism = sc.defaultParallelism

    num_bytes = sum_bytes(dataframe, bytes_column)

    # Cache
    converter = make_spark_converter(
        dataframe, parquet_row_group_size_bytes=int(num_bytes / parallelism)
    )

    return converter


def data_augmentation(rotation_angle: int = 15, 
                      prob_H_flip: float = 0.2, 
                      prob_V_flip: float = 0.2, 
                      prob_blur: [float, float] = [0.0, 0.1]) -> list:
    """
    Data Augmentation function to add label invariant transforms to training pipeline
    Applies a series of transformations such as rotation, horizontal and vertical flips, and Gaussian blur to each image
    """
    transforms = [
        v2.ToImage(),
        v2.RandomRotation(rotation_angle),
        v2.RandomHorizontalFlip(prob_H_flip),
        v2.RandomVerticalFlip(prob_V_flip),
        v2.GaussianNoise(**prob_blur),
    ]
    return transforms

