"""
This module contains low-level functions that transform column objects to a column object. These functions are utility-like but specific to preprocessing tasks.
"""
import io
from typing import Union

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



def sum_column(dataframe, column: "str") -> Union[int, float]:
    """
    Returns the sum of a column in a dataframe.

    Args:
        dataframe: DataFrame
        column: Numeric column or column name containing numeric values.
    """
    result = dataframe.select(F.sum(column)).first()
    return result[0]


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