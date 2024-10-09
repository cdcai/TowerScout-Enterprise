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
        [f"('{x}', '{y}', '{json.dumps(z)}')" for (x, y, z) in validated_data]
    )
    uuids = ", ".join([f"'{x}'" for (x, y, z) in validated_data])

    connection = sql.connect(
        server_hostname="<host-name>", http_path="<http-path>", access_token="<token>"
    )

    cursor = connection.cursor()

    query = "DROP VIEW IF EXISTS gold_updates;"
    cursor.execute(query)

    # create temp view containing validated data 
    # perform a join with silver table on uuid to get relevant information for image from silver table
    # using from_json to unpack bounding boxes
    query = f"""
            CREATE TEMPORARY VIEW gold_updates AS
            WITH temp_data(uuid, imgHash, bboxs) AS (
            VALUES
                {values}
            )

            SELECT from_json(temp.bboxs, 'bboxs array<array<float>>') as bbox, temp.uuid, temp.imgHash, silver.requestId, silver.userId, silver.imagePath
            FROM {catalog}.{schema}.{silver_table} AS silver
            JOIN temp_data AS temp
            ON silver.uuid = temp.uuid
            WHERE silver.path in ({uuids});
            """
    cursor.execute(query)

    # merge temp view into gold table on image hash
    query = f"""
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
    cursor.execute(query)

    cursor.close()
    connection.close()