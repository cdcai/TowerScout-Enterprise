import pytest
from pyspark.sql import SparkSession, DataFrame
from tsdb.preprocessing.transformations import compute_bytes
from pyspark.sql.types import StructType, StructField, BinaryType
from pyspark.sql import Row
from typing import List, Tuple

@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """
    Simple SparkSession fixture to test basic functionality.
    """
    spark_session = (
        SparkSession.builder.master("local")
        .appName("test_compute_bytes")
        .getOrCreate()
    )
    return spark_session

@pytest.fixture(scope="module")
def binary_data_df(spark: SparkSession) -> DataFrame:
    """
    Creates a reusable DataFrame with binary data for tests.
    """
    data: List[Tuple[str]] = [
        ("towerscout",), 
        ("world",), 
        ("abcdef",), 
        (None,)
    ]
    schema: List[str] = ["binary_column"]
    return spark.createDataFrame(data, schema)

def test_compute_bytes(spark: SparkSession, binary_data_df: DataFrame) -> None:
    """
    Unit test for the compute_bytes function.
    """
    # Apply compute_bytes
    result_df: DataFrame = compute_bytes(binary_data_df, "binary_column", "bytes")
    result: List[Row] = result_df.collect()

    # Expected results
    expected: List[Tuple[str, int]] = [
        ("towerscout", 14),  
        ("world", 9), 
        ("abcdef", 10), 
        (None, None),
    ]

    # Validate results
    assert [(row.binary_column, row.bytes) for row in result] == expected, \
        f"Expected {expected}, but got {[(row.binary_column, row.bytes) for row in result]}"

def test_compute_bytes_empty_column(spark: SparkSession) -> None:
    """
    Test compute_bytes with an empty DataFrame.
    """
    schema: StructType = StructType([StructField("binary_column", BinaryType(), True)])
    empty_df: DataFrame = spark.createDataFrame([], schema=schema)
    result: DataFrame = compute_bytes(empty_df, "binary_column")
    assert result.count() == 0, f"Expected empty DataFrame, but got {result.collect()}"

def test_compute_bytes_null_only_column(spark: SparkSession) -> None:
    """
    Test compute_bytes with a DataFrame containing only nulls in the binary column.
    """
    schema: StructType = StructType([StructField("binary_column", BinaryType(), True)])
    null_df: DataFrame = spark.createDataFrame([(None,), (None,), (None,)], schema=schema)
    result: DataFrame = compute_bytes(null_df, "binary_column")

    # Check if the result is a DataFrame and contains None for bytes in all rows
    result_data: List[Row] = result.collect()
    expected_data: List[Row] = [
        Row(binary_column=None, bytes=None),
        Row(binary_column=None, bytes=None),
        Row(binary_column=None, bytes=None),
    ]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"

def test_compute_bytes_large_binary_data(spark: SparkSession) -> None:
    """
    Test compute_bytes with a column containing large binary strings.
    """
    data: List[Tuple[str]] = [
        ("a" * 100,),
        ("b" * 1000,),
        ("c" * 10000,),
    ]
    schema: List[str] = ["binary_column"]
    large_df: DataFrame = spark.createDataFrame(data, schema)
    result_df: DataFrame = compute_bytes(large_df, "binary_column", "bytes")
    result: List[Row] = result_df.collect()

    # Expect base bytes + length of each string
    expected: List[Tuple[str, int]] = [
        ("a" * 100, 104),
        ("b" * 1000, 1004),
        ("c" * 10000, 10004),
    ]
    assert [(row.binary_column, row.bytes) for row in result] == expected, \
        f"Unexpected result for large binary data: {result}"
