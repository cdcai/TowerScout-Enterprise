"""
This module contains DataFrame -> Dataframe transformations that are used in PySpark transform chains
"""
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


def train_test_val_split(
    dataframe: DataFrame,
    train_ratio: float|int,
    test_ratio: float|int,
    val_ratio: float|int,
    seed: int=42
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Splits a Spark dataframe into train, test, and validation sets using the provided ratios.
    The ratios can be floats or integers, however; if the values do not sum to 1.0, randomSplit
    will normalize them. The seed is used to ensure reproducibility.

    Args:
        dataframe (DataFrame): Input dataframe to be split.
        train_ratio (float or int): The ratio of the train set.
        test_ratio (float or int): The ratio of the test set.
        val_ratio (float or int): The ratio of the validation set.
        seed (int): The seed to use for randomSplit reproducibility

    Returns:
        tuple: A tuple containing the train, test, and validation dataframes.
    """
    train_df, test_df, val_df = (
        dataframe
        .randomSplit([train_ratio, test_ratio, val_ratio], seed=seed)
    )
    return train_df, test_df, val_df


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
