"""
This module contains low-level functions that transform column objects to a column object. These functions are utility-like but specific to preprocessing tasks.
"""
import io

import numpy as np
import pandas as pd

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as pst

from PIL import Image, ImageStat

statistics_schema = pst.StructType([
    pst.StructField("mean", pst.ArrayType(pst.DoubleType())),
    pst.StructField("median", pst.ArrayType(pst.IntegerType())),
    pst.StructField("stddev", pst.ArrayType(pst.DoubleType())),
    pst.StructField("extrema", pst.ArrayType(pst.ArrayType(pst.IntegerType()))),
])


def image_statistics_udf(image_binary: pst.BinaryType) -> statistics_schema:
    """
    Returns a struct containing the mean, median, stddev, and extrema of an image binary

    Args:
        image_binary: Image encoded as binary values
    """
    image = Image.open(io.BytesIO(image_binary))
    image_statistics = ImageStat.Stat(image)

    return {
        "mean": image_statistics.mean,
        "median": image_statistics.median,
        "stddev": image_statistics.stddev,
        "extrema": image_statistics.extrema,
    }


def compute_image_statistics(dataframe: DataFrame, image_column: str) -> DataFrame:
    """
    Returns a dataframe with column of computed image statistics

    Args:
        dataframe: DataFrame
        image_column: Name of column in dataframe that contains image binaries
    
    TODO: Determine if this is the best place for this function
    """
    return dataframe.withColumn("statistics", F.expr(f"image_statistics_udf({image_column})"))


@F.pandas_udf(returnType="array<array<integer>>")
def sum_arrays(arrays: pd.Series) -> np.ndarray:
    """
    Sums all the arrays in the input Series. All arrays must be of same shape.
    The return type hint `np.ndarray` indicates that the function returns 
    a numpy array. This function is used to perform a grouped aggregation 
    on a column containing 2D arrays. 
    """
    return arrays.sum(axis=0)