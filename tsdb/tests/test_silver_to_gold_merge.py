import pytest
import json
from typing import Any

from pyspark.sql import SparkSession

from tsdb.utils.silver_to_gold import (
    create_updates_view_query,
    create_gold_merge_query,
    convert_data_to_str,
)


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """
    Returns the SparkSession to be used in tests that require Spark
    """
    spark = (
        SparkSession.builder.master("local")
        .appName("test_silver_to_gold_functions")
        .getOrCreate()
    )

    return spark


@pytest.fixture()
def db_args() -> list[str]:
    return ["edav_dev_csels", "towerscout", "test_image_silver", "test_image_gold"]


@pytest.fixture()
def validated_data() -> dict[str, Any]:
    validated = (
        (
            "75c96459-1946-4950-af0f-df774c6b1f52_tmp1sdnfexw0",
            -1467659206,
            [
                {"conf": 0.77, "class": 0, "x1": 8.2, "x2": 5.4, "y1": 5.1, "y2": 9.2},
                {"conf": 0.88, "class": 0, "x1": 0.2, "x2": 3.3, "y1": 55.4, "y2": 3.5},
            ],
        ),
        (
            "86759e8c-2ac3-4458-a480-16e391bf3742_tmp1sdnfexw0",
            "802091180",
            [
                {
                    "conf": 0.8,
                    "class": 0,
                    "x1": 1.2,
                    "x2": 75.4,
                    "y1": 55.1,
                    "y2": 98.2,
                },
                {"conf": 0.9, "class": 0, "x1": 1.2, "x2": 2.3, "y1": 3.4, "y2": 7.5},
            ],
        ),
    )

    return validated


def test_create_update_view_query(
    spark: SparkSession, validated_data: dict[str, Any], db_args: dict[str, str]
) -> None:
    """
    Tests the create_updates_view_query function
    """

    catalog, schema, silver_table, _ = db_args
    values, uuids = convert_data_to_str(validated_data)

    spark.sql("DROP VIEW IF EXISTS gold_updates;")

    create_updates_view = create_updates_view_query(
        catalog, schema, silver_table, values, uuids
    )

    spark.sql(create_updates_view)

    assert spark.catalog.tableExists("gold_updates")


def test_merge_updates_into_gold(
    spark: SparkSession, validated_data: dict[str, Any], db_args: dict[str, str]
) -> None:
    """
    Tests the create_gold_merge_query function
    """

    catalog, schema, silver_table, gold_table = db_args

    # remove existing gold_table updates view if it exists
    spark.sql("DROP VIEW IF EXISTS gold_updates;")

    # remove testing data from gold table if it exists
    query = f"DELETE FROM {catalog}.{schema}.{gold_table} WHERE image_hash IN (-1467659206, 802091180);"
    spark.sql(query)

    values, uuids = convert_data_to_str(validated_data)

    create_updates_view = create_updates_view_query(
        catalog, schema, silver_table, values, uuids
    )

    spark.sql(create_updates_view)

    merge_updates_into_gold = create_gold_merge_query(catalog, schema, gold_table)

    spark.sql(merge_updates_into_gold)

    query = f"SELECT * FROM {catalog}.{schema}.{gold_table} WHERE image_hash IN (-1467659206, 802091180);"
    df = spark.sql(query)

    assert df.count() == 2