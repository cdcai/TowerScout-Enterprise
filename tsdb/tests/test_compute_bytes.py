import pytest
from pyspark.sql import SparkSession, DataFrame
from tsdb.preprocessing.transformations import compute_bytes
from pyspark.sql.types import StructType, StructField, BinaryType
from pyspark.sql import Row


@pytest.fixture(scope="module")
def my_spark_session() -> SparkSession:
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
def binary_data_df(my_spark_session) -> DataFrame:
    """
    Creates a reusable DataFrame with binary data for tests.
    """
    data = [("towerscout",), ("world",), ("abcdef", ), (None,)]
    schema = ["binary_column"]
    return my_spark_session.createDataFrame(data, schema)

def test_compute_bytes(my_spark_session, binary_data_df):
    """
    Unit test for the compute_bytes function.
    """
    # Apply compute_bytes
    result_df = compute_bytes(binary_data_df, "binary_column", "bytes")
    result = result_df.collect()

    # Expected results
    expected = [
        ("towerscout", 14),  
        ("world", 9), 
        ("abcdef",10), 
        (None, 4),     
    ]

    # Validate results
    assert [(row.binary_column, row.bytes) for row in result] == expected


def test_compute_bytes_empty_column(my_spark_session):
    """
    Test compute_bytes with an empty DataFrame.
    """
    schema = StructType([StructField("binary_column", BinaryType(), True)])
    empty_df = my_spark_session.createDataFrame([], schema=schema)
    result = compute_bytes(empty_df, "binary_column")
    assert result.count() == 0, f"Expected empty DataFrame, but got {result.collect()}"


def test_compute_bytes_null_only_column(my_spark_session):
    """
    Test compute_bytes with a DataFrame containing only nulls in the binary column.
    """
    schema = StructType([StructField("binary_column", BinaryType(), True)])
    null_df = my_spark_session.createDataFrame([(None,), (None,), (None,)], schema=schema)
    result = compute_bytes(null_df, "binary_column")
    # Check if the result is a DataFrame and contains 0 for all rows
    result_data = result.collect()
    expected_data = [Row(binary_column=None, bytes=4),
                     Row(binary_column=None, bytes=4),
                     Row(binary_column=None, bytes=4)]
    assert result_data == expected_data, f"Expected {expected_data}, but got {result_data}"


def test_compute_bytes_large_binary_data(my_spark_session):
    """
    Test compute_bytes with a column containing large binary strings.
    """
    data = [("a" * 100, ), ("b" * 1000, ), ("c" * 10000, )]
    schema = ["binary_column"]
    large_df = my_spark_session.createDataFrame(data, schema)
    result_df = compute_bytes(large_df, "binary_column", "bytes")
    result = result_df.collect()

    # Expect base bytes + length of each string
    expected = [
        ("a" * 100, 104),
        ("b" * 1000, 1004),
        ("c" * 10000, 10004),
    ]
    assert [(row.binary_column, row.bytes) for row in result] == expected, \
        f"Unexpected result for large binary data: {result}"

