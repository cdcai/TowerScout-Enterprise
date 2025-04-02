""""This module contains unit tests for the functions in the ml/drift.py module"""
from typing import Any
from datetime import datetime, timedelta

import pytest
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ArrayType,
    TimestampType,
    IntegerType,
)

from tsdb.ml.drift import get_struct_counts, compute_drift_score


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
def timestamp_col() -> str:
    return "test_timestamp"


@pytest.fixture(scope="module")
def array_struct_col() -> str:
    return "animal_labels"


@pytest.fixture(scope="module")
def filter_clause() -> str:
    return "x.class = 'cat' and x.label = 1"


@pytest.fixture(scope="module")
def time_window_days() -> int:
    return 2


@pytest.fixture(scope="module")
def example_df(spark, timestamp_col: str, array_struct_col: str) -> DataFrame:

    # Define the schema for the DataFrame
    schema = StructType([
        StructField(timestamp_col, TimestampType(), True),
        StructField(array_struct_col, ArrayType(StructType([
            StructField("class", StringType(), True),
            StructField("label", IntegerType(), True)
        ])), True)
    ])

    # Create sample data with specific timestamps
    data = [
        (datetime.now(), [{"class": "dog", "label": 0}, {"class": "cat", "label": 1}]),
        (datetime.now() - timedelta(days=1), [{"class": "dog", "label": 0}]),
        (datetime.now() - timedelta(days=30), [{"class": "dog", "label": 0}, 
                                               {"class": "dog", "label": 0}, 
                                               {"class": "cat", "label": 1}])
    ]

    df = spark.createDataFrame(data, schema)

    return df


def test_get_struct_counts(example_df: DataFrame, timestamp_col: str, array_struct_col: str, filter_clause: str, time_window_days: int) -> None:

    new_df = get_struct_counts(example_df, timestamp_col, array_struct_col, filter_clause, time_window_days)

    assert new_df.count() == 2, f"Expected 2 rows, got {new_df.count()} rows. Only 1 record should have been filtered out by the time window range."
    assert new_df.agg({"num_filtered_structs": "sum"}).collect()[0][0] == 1, "Expected 1 struct containing a cat to be in the filtered dataframe."
    assert new_df.agg({"num_structs": "sum"}).collect()[0][0] == 3, "Expected 3 structs total to be in the filtered dataframe."


def test_compute_drift_score(example_df: DataFrame, timestamp_col: str, array_struct_col: str, filter_clause: str, time_window_days: int) -> None:
    
    new_df = get_struct_counts(example_df, timestamp_col, array_struct_col, filter_clause, time_window_days)
    drift_score = compute_drift_score(new_df, "num_filtered_structs", "num_structs")

    absolute_tolerance = 1e-4
    assert pytest.approx(drift_score, abs=absolute_tolerance) == 1/3, "Expected drift score to be 1/3"
