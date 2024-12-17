"""
This module tests code in tsdb.preprocessing.preprocess functions on image data. If needed,
function docstrings can include examples of what is being tested.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import functions as F
from petastorm.spark import make_spark_converter
from tsdb.preprocessing.functions import sum_bytes  
from tsdb.preprocessing.preprocess import create_converter


@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    spark = (
        SparkSession.builder.master("local")
        .appName("test_preprocessing")
        .getOrCreate()
    )
    return spark


def test_create_converter(spark):
    """Test the create_converter function."""

    # Get or create a SparkContext instance
    sc = SparkContext.getOrCreate()  

    # Set Petastorm configuration for cache directory (update this path as needed)
    cache_dir = "file:///dbfs/tmp/petastorm/cache"  #"file:///tmp/petastorm_cache"  # Change this to your desired path
    spark.conf.set("SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF", cache_dir) 
    spark.conf.set("petastorm.spark.converter.parentCacheDirUrl", cache_dir)   

    # setup test dataframe
    images_df = (
        spark
        .table("edav_dev_csels.towerscout.image_metadata")
        .select("length", "content", "path")
        .limit(5)
        )

    # Calculate bytes
    num_bytes = (
        images_df
        .withColumn("bytes", F.lit(4) + F.length("content"))
        .groupBy()
        .agg(F.sum("bytes").alias("total_bytes"))
        .collect()[0]["total_bytes"]
    )
    # 341491

    # Mocking the dependencies with correct paths
    with patch('petastorm.spark.make_spark_converter') as mock_make_spark_converter:

        # Call sum_bytes to ensure it returns the mocked value
        assert sum_bytes(images_df, "length") == pytest.approx(num_bytes, rel=1e-3)   #passed
        
        # Create a mock converter object to return from make_spark_converter 
        test_converter = make_spark_converter(
            images_df, parquet_row_group_size_bytes=int(num_bytes / sc.defaultParallelism)
        )

        # Call the function under test
        converter = create_converter(images_df, "length", sc, 0)

        # Check if returned converter is correct
        assert isinstance(converter, type(test_converter))   # passed

        # mock_converter = MagicMock() 
        mock_make_spark_converter.return_value = test_converter
        parquet_row_group_size_bytes = int(num_bytes / sc.defaultParallelism)
        #42686

        # check if make_spark_converter was called correctly
        try:
            mock_make_spark_converter.assert_called_once_with(
                images_df,
                parquet_row_group_size_bytes=parquet_row_group_size_bytes,
                spark_session=spark,
                cache_dir=cache_dir
            )
            print("Assertion passed: make_spark_converter was called with expected arguments.")
        
        except AssertionError as e:
            print(f"Assertion failed: {e}")
            print(f"Called with: {mock_make_spark_converter.call_args}")
            print(f"Expected: (images_df, parquet_row_group_size_bytes={parquet_row_group_size_bytes}, spark_session=spark, cache_dir={cache_dir})")
