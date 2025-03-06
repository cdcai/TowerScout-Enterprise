import json
import requests
from typing import Any

from numpy.random import choice


DATABRICKS_INSTANCE = "adb-1881246389460182.2.azuredatabricks.net"
PERSONAL_ACCESS_TOKEN = "dapi51d5af94736bbfdfaa7cb944c76cc531-3"
WAREHOUSE_ID = "8605a48953a7f210"
SQL_STATEMENTS_ENDPOINT = f'https://{DATABRICKS_INSTANCE}/api/2.0/sql/statements'


def execute_query(query: str, cursor, connection) -> None:
    """
    Function to execute queries on databricks using the databricks SQL API

    Args:
        query (str): The query to execute
        cursor (sql.Cursor): The cursor to execute the query on
        connection (sql.Connection): The connection to execute the query on
    """
    try:
        cursor.execute(query)
        return
    except:
        cursor.close()
        connection.close()
        return


def convert_data_to_str(
    validated_data: tuple[tuple[str, str, dict[str, Any]]]
) -> tuple[str, str]:
    """
    Function to convert validated data to string for SQL query

    Args:
        validated_data: uuids, image hash and bounding boxes of the image (in that order) validated by an end user
    
    Returns:
        values: the validated data combined into a string
        uuids: the uuids from the validated data combined into a string
    """

    values = ", ".join(
        [
            f"('{uuid}', '{image_hash}', '{json.dumps(bboxes)}', '{choice(a=['train', 'val', 'test'], p=[0.8, 0.1, 0.1])}')"
            for (uuid, image_hash, bboxes) in validated_data
        ]
    )

    uuids = ", ".join([f"'{uuid}'" for (uuid, image_hash, bboxes) in validated_data])

    return values, uuids


def create_gold_table_update_query(values: str, uuids: str) -> str:
    """
    Function to create SQL query string that creates a temp view containing validated
    data to be merged into the gold table.
    The query first extracts all rows from the silver table, via a join, which
    have uuids matching those supplied to the function (uuids of the images being promoted from silver to gold)
    and overrides their bbox field with the bboxes supplied to the function.
    These modified rows are then merged into the gold table on the image hash.
    
    Args:
        values: The values to be inserted into the temp view
        uuids: The uuids to join and filter on
    Returns:
        create_updates_view: SQL query string to create a temp view containing validated data to be merged into the gold table
    """

    gold_table_update_query = f"""
            WITH temp_data(uuid, image_hash, bboxes, split_label) AS 
            (VALUES {values})
            
            MERGE INTO IDENTIFIER(:gold_table) AS target
            USING (
                SELECT 
                    FROM_JSON(temp.bboxes, 'array<struct<`class_name`:string,`secondary`:float,`class`:int,`x1`:float,`y1`:float,`x2`:float,`y2`:float,`conf`:float>>') as bboxes,
                    temp.uuid,
                    temp.image_hash,
                    silver.request_id,
                    silver.user_id,
                    silver.image_path,
                    temp.split_label
                FROM IDENTIFIER(:silver_table) AS silver
                JOIN temp_data AS temp ON silver.uuid = temp.uuid
                WHERE silver.uuid IN ({uuids})
            ) AS source
            
            ON (target.image_hash = source.image_hash)
            WHEN MATCHED THEN
                UPDATE SET 
                    target.bboxes = source.bboxes,
                    target.uuid = source.uuid,
                    target.image_hash = source.image_hash,
                    target.image_path = source.image_path,
                    target.request_id = source.request_id,
                    target.user_id = source.user_id,
                    target.reviewed_time = CURRENT_TIMESTAMP(),
                    target.split_label = source.split_label 
            WHEN NOT MATCHED THEN
                INSERT (bboxes, uuid, image_hash, image_path, request_id, reviewed_time, user_id, split_label) 
                VALUES (source.bboxes, source.uuid, source.image_hash, source.image_path, source.request_id, CURRENT_TIMESTAMP(), source.user_id, source.split_label);
            """

    return gold_table_update_query


def promote_silver_to_gold(
    validated_data: tuple[str, str, dict[int, list[float]]],
    catalog: str,
    schema: str,
    gold_table: str,
    silver_table: str,
) -> dict[str, Any]:
    """
    Function to promote silver data to gold by updating the gold table
    Note: Values and uuids are not parameterized using the 'parameters' arguement
    to the Databricks SQL API.

    Args:
        validated_data: uuids, image hash and bounding boxes of the image (in that order) validated by an end user
        catalog: the catalog of the gold table
        schema: the schema of the gold table
        gold_table: the name of the gold table
        silver_table: the name of the silver table
    
    Returns:
        The response from the Databricks SQL API
    """

    values, uuids = convert_data_to_str(validated_data)

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

    response = requests.post(SQL_STATEMENTS_ENDPOINT, json=data, headers=headers)

    return response.json()