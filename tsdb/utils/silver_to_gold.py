from databricks import sql
import json
from typing import Any

from numpy.random import choice


def execute_query(query: str, cursor: sql.Cursor, connection: sql.Connection) -> None:
    """
    Function to execute queries on databricks using the databricks SQL API

    Args:
        query: The query to execute
        cursor: The cursor to execute the query on
        connection: The connection to execute the query on
    """
    try:
        cursor.execute(query)
        return
    except:
        cursor.close()
        connection.close()
        return

def convert_data_to_str(validated_data: tuple[str, str, dict[str, Any]]) -> tuple[str, str]:
    """
    Function to convert validated data to string for SQL query

    Args:
        validated_data: uuids, image hash and bounding boxes of the image (in that order) validated by an end user
    """

    values = ", ".join(
        [
            f"('{uuid}', '{image_hash}', '{json.dumps(bboxes)}', '{choice(a=['train', 'val', 'test'], p=[0.6, 0.2, 0.2])}')"
            for (uuid, image_hash, bboxes) in validated_data
        ]
    )

    uuids = ", ".join([f"'{uuid}'" for (uuid, image_hash, bboxes) in validated_data])

    return values, uuids

def create_updates_view_query(
    catalog: str, schema: str, silver_table: str, values: str, uuids: str
) -> str:
    """
    Function to create SQL query string that creates a temp view containing validated
    data to be merged into the gold table.
    This is done by performing a join with silver table on uuid to
    get relevant information for image from silver table using from_json to unpack bounding boxes
    Args:
        catalog: The catalog to use
        schema: The schema to use
        silver_table: The name of the silver table
        values: The values to be inserted into the temp view
        uuids: The uuids to join and filter on
    Returns:
        create_updates_view: SQL query string to create a temp view containing validated data to be merged into the gold table
    """

    create_updates_view = f"""
        CREATE TEMPORARY VIEW gold_updates AS
        WITH temp_data(uuid, image_hash, bboxes, split_label) AS (
        VALUES
            {values}
        )
        
        SELECT from_json(temp.bboxes, 'array<struct<`class`:int,`x1`:float,`y1`:float,`x2`:float,`y2`:float,`conf`:float>>') as bboxes, temp.uuid, temp.image_hash, silver.request_id, silver.user_id, silver.image_path, temp.split_label
        FROM {catalog}.{schema}.{silver_table} AS silver
        JOIN temp_data AS temp
        ON silver.uuid = temp.uuid
        WHERE silver.uuid in ({uuids});
        """

    return create_updates_view


def create_gold_merge_query(catalog: str, schema: str, gold_table: str) -> str:
    """
    Function to create SQL query string that creates a temp view containing validated
    data to be merged into the gold table.
    This is done by performing a join with silver table on uuid to
    get relevant information for image from silver table using from_json to unpack bounding boxes
    Args:
        catalog: The catalog to use
        schema: The schema to use
        silver_table: The name of the gold table to merge the temp view 'gold_updates' into
    Returns:
        merge_updates_into_gold: SQL merge query string
    """

    merge_updates_into_gold = f"""
        MERGE INTO {catalog}.{schema}.{gold_table} AS target
        USING gold_updates AS source
        ON (target.image_hash = source.image_hash)
        WHEN MATCHED THEN
            UPDATE SET target.bboxes = source.bboxes,
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

    return merge_updates_into_gold


def promote_silver_to_gold(
    validated_data: tuple[str, str, dict[int, list[float]]],
    catalog: str,
    schema: str,
    gold_table: str,
    silver_table: str,
) -> None:
    """
    Function to promote silver data to gold by updating the gold table
    validated_data: uuids, image hash and bounding boxes of the image (in that order) validated by an end user
    catalog: the catalog of the gold table
    schema: the schema of the gold table
    gold_table: the name of the gold table
    silver_table: the name of the silver table
    """

    # create connection to the SQL warehouse
    connection = sql.connect(
        server_hostname="<host-name>", http_path="<http-path>", access_token="<token>"
    )

    values, uuids = convert_data_to_str(validated_data)

    cursor = connection.cursor()

    drop_existing_view = "DROP VIEW IF EXISTS gold_updates;"

    execute_query(drop_existing_view, cursor, connection)

    create_updates_view = create_updates_view_query(
        catalog, schema, silver_table, values, uuids
    )

    execute_query(create_updates_view, cursor, connection)

    # merge temp view into gold table on image hash
    merge_updates_into_gold = create_gold_merge_query(catalog, schema, gold_table)

    execute_query(merge_updates_into_gold, cursor, connection)

    cursor.close()
    connection.close()