"""
This module tests code in tsdb.preprocessing.functions that are transformation based. If needed,
function docstrings can include examples of what is being tested.
"""
import pytest
import tsdb.preprocessing.utils as putils

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """
    Returns the SparkSession to be used in tests that require Spark
    """
    spark = (
        SparkSession.builder.master("local")
        .appName("test_functions_transformation")
        .getOrCreate()
    )
    
    return spark


def test_cast_to_column_string_input(spark: SparkSession) -> None:
    """
    Tests the cast_to_column function with a string input and verifies it returns
    a column object. Column objects are oftenmore useful than strings but we still
    want the flexibility of being able to pass a column name as an input.
    """
    col_name = "test_column"
    test_dataframe = spark.createDataFrame([(1,), (2,), (3,)], [col_name])
    
    result = putils.cast_to_column(col_name)

    # Result should be a PySpark Column object
    assert isinstance(result, F.Column), f"Expected a PySpark Column object, got {type(result)}"
