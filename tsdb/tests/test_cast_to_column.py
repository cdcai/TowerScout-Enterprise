import pytest
from pyspark.sql import SparkSession, Column
import pyspark.sql.functions as F
from tsdb.preprocessing.utils import cast_to_column

@pytest.fixture(scope="module")
def spark_session() -> SparkSession:
    """
    Provides a SparkSession for testing.
    """
    spark_session = (
        SparkSession.builder.master("local")
        .appName("test_cast_to_column")
        .getOrCreate()
    )
    return spark_session

def test_cast_to_column_with_string():
    """
    Test that cast_to_column correctly converts a string input to a PySpark Column.
    """
    col_name = "test_column"
    result = cast_to_column(col_name)
    assert isinstance(result, Column), f"Expected a Column, but got {type(result)}"


def test_cast_to_column_with_column():
    """
    Test that cast_to_column returns the same Column object if a Column is passed.
    """
    input_column = F.col("test_column")
    result = cast_to_column(input_column)
    assert result is input_column, "cast_to_column should return the same Column object if input is already a Column."
    
def test_cast_to_column_with_empty_string():
    """
    Test how cast_to_column handles an empty string input and ensure it raises an error.
    """
    result = cast_to_column("")
    assert isinstance(result, Column), f"Expected Column, but got {type(result)}"
    assert str(result) == "Column<''>", f"Unexpected result: {result}"


def test_cast_to_column_with_invalid_type():
    """
    Test how cast_to_column handles an invalid input type by ensuring
    it raises an error if the type is not str or Column.
    """
    invalid_input = 12345  # Invalid type (integer)
    result = cast_to_column(invalid_input)
    assert isinstance(result, int), f"Expected int for invalid input, but got {type(result)}"
    assert result == invalid_input, f"Expected {invalid_input}, but got {result}"

