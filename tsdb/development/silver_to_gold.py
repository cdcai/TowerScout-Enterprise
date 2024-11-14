from databricks import sql
import json


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

    values = ", ".join(
        [f"('{uuid}', '{image_hash}', '{json.dumps(bboxes)}')" for (uuid, image_hash, bboxes) in validated_data]
    )
    uuids = ", ".join([f"'{uuid}'" for (uuid, image_hash, bboxes) in validated_data])

    connection = sql.connect(
        server_hostname="<host-name>", http_path="<http-path>", access_token="<token>"
    )

    cursor = connection.cursor()

    drop_existing_view = "DROP VIEW IF EXISTS gold_updates;"
    try:
        cursor.execute(drop_existing_view)
    except:
        cursor.close()
        connection.close()
        return 

    # create temp view containing validated data 
    # perform a join with silver table on uuid to get relevant information for image from silver table
    # using from_json to unpack bounding boxes
    create_updates_view = f"""
            CREATE TEMPORARY VIEW gold_updates AS
            WITH temp_data(uuid, image_hash, bboxes) AS (
            VALUES
                {values}
            )

            SELECT from_json(temp.bboxes, 'bboxes array<struct<label:int,x1:float,y1:float,x2:float,y2:float,conf:float>>') as bboxes, temp.uuid, temp.image_hash, silver.request_id, silver.user_id, silver.image_path
            FROM {catalog}.{schema}.{silver_table} AS silver
            JOIN temp_data AS temp
            ON silver.uuid = temp.uuid
            WHERE silver.uuid in ({uuids});
            """
    try:
        cursor.execute(create_updates_view)
    except:
        cursor.close()
        connection.close()
        return 

    # merge temp view into gold table on image hash
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
                        target.reviewed_time = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN
                INSERT (bboxes, uuid, image_hash, image_path, request_id, reviewed_time, user_id) 
                VALUES (source.bboxes, source.uuid, source.image_hash, source.image_path, source.request_id, CURRENT_TIMESTAMP() source.user_id);
            """
    try:
        cursor.execute(merge_updates_into_gold)
    except:
        cursor.close()
        connection.close()
        return

    cursor.close()
    connection.close()