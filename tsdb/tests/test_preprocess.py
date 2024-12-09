"""
This module tests code in tsdb.preprocessing.preprocess functions on image data. If needed,
function docstrings can include examples of what is being tested.
"""

import pytest
from unittest.mock import MagicMock, patch
from pyspark.sql import SparkSession
from pyspark import SparkContext
from tsdb.preprocessing.preprocess import data_augmentation
from tsdb.preprocessing.preprocess import create_converter
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from tsdb.preprocessing.functions import sum_bytes


# Assuming these are imported from your module where create_converter is defined
# from your_module import create_converter, sum_bytes, make_spark_converter


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """
    Returns the SparkSession to be used in tests that require Spark
    """
    spark = (
        SparkSession.builder.master("local")
        .appName("test_data_processing")
        .getOrCreate()
    )
    
    return spark


@pytest.fixture(scope="module")
def image_df(spark) -> DataFrame:
    data = [
        ("img0", "bbox0"),
        ("img1", "bbox1"),
        ("img2", "bbox2"),
        ("img3", "bbox3"),
        ("img4", "bbox4"),
        ("img5", "bbox5"),
        ("img6", "bbox6"),
        ("img7", "bbox7"),
        ("img8", "bbox8"),
        ("img9", "bbox9"),
    ]

    return spark.createDataFrame(data, ["image", "bbox"])


def test_data_augmentation() -> None:
    transforms = data_augmentation()
    assert isinstance(transforms, list), f"Expected a list of transforms got {type(transforms)}"


def test_create_converter(spark, image_df):
    # Create a mock dataframe
    data = [(1,), (2,), (3,)]
    df = spark.createDataFrame(data, ["bytes_column"])
    # df = image_df.withColumn("bytes_column", F.lit(100))

    # Mocking the dependencies
    with patch('sum_bytes') as mock_sum_bytes, \
         patch('make_spark_converter') as mock_make_spark_converter:
        
        # Set up return values for mocks
        mock_sum_bytes.return_value = 300  # Example byte count
        
        # Create a mock converter object to return from make_spark_converter
        mock_converter = MagicMock()
        mock_make_spark_converter.return_value = mock_converter
        
        sc = SparkContext.getOrCreate()  # Get or create a SparkContext instance
        
        # Call the function under test
        converter = create_converter(df, "bytes_column", sc)

        # Assertions to ensure everything works as expected
        assert converter == mock_converter  # Check if returned converter is correct
        
        # Check if sum_bytes was called correctly
        mock_sum_bytes.assert_called_once_with(df, "bytes_column")
        
        # Check if make_spark_converter was called with correct parameters
        assert len(mock_make_spark_converter.call_args) > 0  # Ensure it was called at least once

        parquet_row_group_size_bytes = int(300 / sc.defaultParallelism)
        
        assert (
            mock_make_spark_converter.call_args[0][1] == parquet_row_group_size_bytes
            )  # Check
        
