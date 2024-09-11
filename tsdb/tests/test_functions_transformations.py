"""
This module tests code in tsdb.preprocessing.functions that are transformation based, aka
"""
import pytest
import tsdb.preprocessing.functions as tsf

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


@pytest.fixture(scope="module")
def spark():
    spark = (
        SparkSession.builder.master("local")
        .appName("test_functions_transformation")
        .getOrCreate()
    )
    
    return spark


def test_cast_to_column_string_input(spark):
    """
    What are you testing?
    Why are you testing
    """
    col_name = "test_column"
    test_dataframe = spark.createDataFrame([(1,), (2,), (3,)], [col_name])
    
    result = tsf.cast_to_column(col_name)

    # This should be a PySpark Column object
    assert isinstance(result, F.Column), f"Expected a Column object, got {type(result)}"
