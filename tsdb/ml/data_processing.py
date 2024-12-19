import io
from functools import partial

import numpy as np

from petastorm import TransformSpec
from petastorm.spark import SparkDatasetConverter, make_spark_converter

import torchvision


from PIL import Image
from pyspark.sql import DataFrame
from pyspark.context import SparkContext
import pyspark.sql.functions as F

from tsdb.preprocessing.utils import cast_to_column
from tsdb.preprocessing.transformations import compute_bytes
from tsdb.preprocessing.preprocess import create_converter


def transform_row(batch_pd): # pragma: nocover
    """
    Defines how to transform partition elements
    """
    transformers = [
        torchvision.transforms.Lambda(lambda image: Image.open(io.BytesIO(image)))
    ]

    transformers.extend(
        [
            torchvision.transforms.Resize(128),
            torchvision.transforms.ToTensor(),
        ]
    )

    transformer_pipeline = torchvision.transforms.Compose(transformers)

    # Needs to be row-major array
    batch_pd["features"] = batch_pd["content"].map(
        lambda image: np.ascontiguousarray(transformer_pipeline(image).numpy())
    )

    return batch_pd[["features"]]


def get_transform_spec(): # pragma: nocover
    """
    Applies transforms across partitions
    """
    spec = TransformSpec(
        partial(transform_row),
        edit_fields=[
            ("features", np.float32, (3, 128, 128), False),
        ],
        selected_fields=["features"],
    )

    return spec



def get_converter(
    cat_name="edav_dev_csels", sch_name="towerscout_test_schema", batch_size=8
):
    petastorm_path = "file:///dbfs/tmp/petastorm/cache"
    images = spark.table(f"{cat_name}.{sch_name}.image_metadata").select(
        "content", "path"
    )

    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, petastorm_path)

    # Calculate bytes
    num_bytes = (
        images.withColumn("bytes", F.lit(4) + F.length("content"))
        .groupBy()
        .agg(F.sum("bytes").alias("bytes"))
        .collect()[0]["bytes"]
    )

    # Cache
    converter = make_spark_converter(
        images, parquet_row_group_size_bytes=int(num_bytes / sc.defaultParallelism)
    )

    context_args = {"transform_spec": get_transform_spec(), "batch_size": 8}

    return converter


def get_converter_df(dataframe: DataFrame, sc: SparkContext) -> callable: # pragma: nocover
    """
    Creates a petastrom converter for a Spark dataframe

    Args:
        dataframe: The Spark dataframe
    Returns:
        callable: A petastorm converter
    """

    dataframe = dataframe.transform(compute_bytes, "content")
    converter = create_converter(dataframe, "bytes", sc)

    return converter


def split_data(images: DataFrame) -> (DataFrame, DataFrame, DataFrame): # pragma: nocover
    """
    Splits a Spark dataframe into train, test, and validation sets.
    Note that the input dataframe must have a column "label" with
    binary values 0 and 1 to function correctly.

    Args:
        df (DataFrame): Input dataframe to be split.

    Returns:
        tuple: A tuple containing the train, test, and validation dataframes.
    """
    # TODO: Combine split_data and split_datanolabel into a single function
    # and make them more robust

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
