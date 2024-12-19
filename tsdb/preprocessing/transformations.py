"""
This module contains DataFrame -> Dataframe transformations that are used in PySpark transform chains
"""
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import tsdb.preprocessing.utils as utils


def compute_bytes(
    dataframe: DataFrame, binary_column: "ColumnOrName", col_name: "str" = "bytes"
) -> DataFrame:
    """
    Returns a dataframe with a bytes column, which calculates the number of
    bytes in a binary column. While this function is intended to be used for
    binary data, we do not strictly enforce this.

    Args:
        dataframe: DataFrame
        binary_column: Name or col that has bit data
        col_name (default="bytes"): The name of the new result column
    """
    base_bytes = 4
    binary_column = utils.cast_to_column(binary_column)
    num_bytes = F.lit(base_bytes) + F.length(binary_column)

    return dataframe.withColumn(col_name, num_bytes)


def perform_inference(dataframe, inference_udf, input_column: str="content"):  # pragma: no cover
    return dataframe.withColumn("results", inference_udf(F.col(input_column)))


def extract_metadata(dataframe, metadata_udf, input_column: str="content"):
    all_metadata = metadata_udf(F.col(input_column))

    image_metadata_keys = (
        "lat",
        "long",
        "width",
        "height"
    )
    image_metadata = F.struct(*[
        all_metadata.getItem(key).alias(key)
        for key in image_metadata_keys
    ])

    return (
        dataframe
        .withColumn("image_metadata", image_metadata)
        .withColumn("image_id", all_metadata.getItem("image_id"))
        .withColumn("map_provider", all_metadata.getItem("map_provider"))
    )

def current_time(dataframe):  # pragma: no cover
    return dataframe.withColumn("processing_time", F.current_timestamp())

def hash_image(dataframe):
    """
    TODO: test this
    """
    return dataframe.withColumn("image_hash", F.hash(F.col("content")))

def parse_file_path(dataframe):
    """
    TODO: test this
    """
    split_col = F.split(F.col("path"), "/")

    user_id = F.element_at(split_col,(-3))
    request_id = F.element_at(split_col,(-2))
    uuid = F.regexp_replace(
        F.element_at(split_col,(-1)),
        pattern=r"\.jpeg",
        replacement=""
    )

    return (
        dataframe
        .withColumn("user_id", user_id)
        .withColumn("request_id", request_id)
        .withColumn("uuid", uuid)
    )
