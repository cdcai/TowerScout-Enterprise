"""
This module tests code in tsdb.preprocessing.preprocess functions on image data. If needed,
function docstrings can include examples of what is being tested.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from pyspark.sql import SparkSession

from tsdb.preprocessing.preprocess import convert_to_mds


@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    spark = (
        SparkSession.builder.master("local").appName("test_preprocessing").getOrCreate()
    )
    return spark
