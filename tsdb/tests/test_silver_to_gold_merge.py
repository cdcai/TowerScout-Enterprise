import pytest
import json
import requests
from typing import Any

from pyspark.sql import SparkSession

from tsdb.utils.silver_to_gold import (
    create_gold_table_update_query,
    convert_data_to_str,
)

# Hardcoding until SP has perimission to access these via secrets
DATABRICKS_INSTANCE = "adb-1881246389460182.2.azuredatabricks.net"
PERSONAL_ACCESS_TOKEN = "dapi51d5af94736bbfdfaa7cb944c76cc531-3"
WAREHOUSE_ID = "8605a48953a7f210"
SQL_STATEMENTS_ENDPOINT = f'https://{DATABRICKS_INSTANCE}/api/2.0/sql/statements'


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
                {
                    "conf": 0.77,
                    "class": 0,
                    "x1": 8.2,
                    "x2": 5.4,
                    "y1": 5.1,
                    "y2": 9.2,
                    "class_name": "ct",
                    "secondary": 1.0,
                },
                {
                    "conf": 0.88,
                    "class": 0,
                    "x1": 0.2,
                    "x2": 3.3,
                    "y1": 55.4,
                    "y2": 3.5,
                    "class_name": "ct",
                    "secondary": 1.0,
                },
            ],
        ),
        (
            "86759e8c-2ac3-4458-a480-16e391bf3742_tmp1sdnfexw0",
            802091180,
            [
                {
                    "conf": 0.8,
                    "class": 0,
                    "x1": 1.2,
                    "x2": 75.4,
                    "y1": 55.1,
                    "y2": 98.2,
                    "class_name": "ct",
                    "secondary": 1.0,
                },
                {
                    "conf": 0.9,
                    "class": 0,
                    "x1": 1.2,
                    "x2": 2.3,
                    "y1": 3.4,
                    "y2": 7.5,
                    "class_name": "ct",
                    "secondary": 1.0,
                },
            ],
        ),
    )

    return validated


def test_create_gold_table_update_query(
    spark: SparkSession, validated_data: dict[str, Any], db_args: dict[str, str]
) -> None:
    """
    Tests the create_gold_merge_query function
    """

    catalog, schema, silver_table, gold_table = db_args

    values, uuids = convert_data_to_str(validated_data)

    # remove testing data from gold table if it exists
    delete_query = f"""DELETE FROM {catalog}.{schema}.{gold_table} 
            WHERE 
            image_hash IN (-1467659206, 802091180) 
            AND 
            uuid IN ("86759e8c-2ac3-4458-a480-16e391bf3742_tmp1sdnfexw0", "75c96459-1946-4950-af0f-df774c6b1f52_tmp1sdnfexw0");
            """
    spark.sql(delete_query)

    gold_table_update_query = create_gold_table_update_query(values, uuids)

    data = {
        "statement": gold_table_update_query,
        "warehouse_id": WAREHOUSE_ID,
        "catalog": catalog,
        "schema": schema,
        "parameters": [
            {"name": "silver_table", "value": silver_table, "type": "STRING"},
            {"name": "gold_table", "value": gold_table, "type": "STRING"},
        ],
    }

    # Set the headers with the personal access token for authentication
    headers = {
        "Authorization": f"Bearer {PERSONAL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(SQL_STATEMENTS_ENDPOINT, json=data, headers=headers).json()

    assert response["status"]["state"] != "FAILED", f"Databricks SQL API returned that the query failed. Response was {response['status']}."

    query = f"""SELECT * FROM {catalog}.{schema}.{gold_table} 
            WHERE 
            image_hash IN (-1467659206, 802091180) 
            AND 
            uuid IN ("86759e8c-2ac3-4458-a480-16e391bf3742_tmp1sdnfexw0", "75c96459-1946-4950-af0f-df774c6b1f52_tmp1sdnfexw0");
            """
    df = spark.sql(query)

    assert df.count() == 2, f"Exactly two records should have been added to the gold table but actual number added was {df.count()}."

    # remove test records once tests are complete 
    spark.sql(delete_query)