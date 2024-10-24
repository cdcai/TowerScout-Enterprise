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
        [f"('{uuid}', '{img_hash}', '{json.dumps(bboxes)}')" for (uuid, img_hash, bboxes) in validated_data]
    )
    uuids = ", ".join([f"'{uuid}'" for (uuid, img_hash, bboxes) in validated_data])

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
            WITH temp_data(uuid, imgHash, bboxs) AS (
            VALUES
                {values}
            )

            SELECT from_json(temp.bboxs, 'bboxs array<array<float>>') as bbox, temp.uuid, temp.imgHash, silver.requestId, silver.userId, silver.imagePath
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
            ON (target.imgHash = source.imgHash)
            WHEN MATCHED THEN
                UPDATE SET target.bboxs = source.bboxs,
                        target.uuid = source.uuid,
                        target.imgHash = source.imgHash,
                        target.imagePath = source.imagePath,
                        target.requestId = source.requestId,
                        target.userId = source.userId,
                        target.reviewedTime = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN
                INSERT (bboxs, uuid, imgHash, imagePath, requestId, reviewedTime) VALUES (source.bboxs, source.uuid, source.imgHash, source.imagePath, source.requestId, source.userId, CURRENT_TIMESTAMP());
            """
    try:
        cursor.execute(merge_updates_into_gold)
    except:
        cursor.close()
        connection.close()
        return

    cursor.close()
    connection.close()