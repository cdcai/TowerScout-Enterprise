# Databricks notebook source
#!pip install databricks-sql-connector

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Create temp view of rows that have matching uuids from the input
# MAGIC 2. Go through and update the bbox's in the temp view to be the ones from the function input
# MAGIC 3. Do a merge (merging on image hash sha1) into the gold table from the temp view
# MAGIC 4. Delete temp view 

# COMMAND ----------

# %sql
# CREATE TABLE edav_dev_csels.towerscout_test_schema.test_image_gold (path STRING, statistics struct<`mean`:array<double>,`median`:array<int>,`stddev`:array<double>,`extrema`:array<array<int>>>);

# COMMAND ----------

catalog = "edav_dev_csels"
schema = "towerscout_test_schema"
gold_table = "test_image_gold"
silver_table = "test_image_silver"


func_input = (
    (
        "abfss://ddphss-csels@davsynapseanalyticsdev.dfs.core.windows.net/PD/TowerScout/Unstructured/test_images/tmp5j0qh5k121.jpg",
        {
            "mean": [-1.2, 2.8, -2.3],
            "median": [6, 6, 6],
            "stddev": [8.0, 8.0, 8.0],
            "extrema": [[5, 1], [2, 21], [6, 55]],
        },
    ),
    (
        "abfss://ddphss-csels@davsynapseanalyticsdev.dfs.core.windows.net/PD/TowerScout/Unstructured/test_images/tmp5j0qh5k115.jpg",
        {
            "mean": [-1.5, 2.6, -2.7],
            "median": [6, 6, 6],
            "stddev": [8.0, 8.0, 8.0],
            "extrema": [[5, 1], [2, 21], [6, 55]],
        },
    ),
)

paths = ", ".join([f"'{x}'" for (x, y) in func_input])

# COMMAND ----------

import json

mappings = ", ".join([f"('{x}', '{json.dumps(y)}')" for (x, y) in func_input])

print(mappings)

# COMMAND ----------

query = "DROP VIEW IF EXISTS gold_updates;"
spark.sql(query)

query = f"""
CREATE TEMPORARY VIEW gold_updates AS
WITH temp_data(path, statistics) AS (
  VALUES
    {mappings}
)

SELECT from_json(temp.statistics, 'mean array<double>, median array<int>, stddev array<double>, extrema array<array<int>>') as statistics, temp.path, silver.length
FROM {catalog}.{schema}.{silver_table} AS silver

JOIN temp_data AS temp
ON silver.path = temp.path
WHERE silver.path in ({paths});
"""

display(spark.sql(query))

# uuid's bounding boxes, image hash, userID, set review time as CURRENT_TIMESTAMP()
# get requestID from silver table

# COMMAND ----------

display(spark.sql("SELECT * from gold_updates"))

# COMMAND ----------

display(spark.sql(f"SELECT * from {catalog}.{schema}.{gold_table}"))

# COMMAND ----------

query = f"""
MERGE INTO {catalog}.{schema}.{gold_table} AS target
USING gold_updates AS source
ON (target.path = source.path)
WHEN MATCHED THEN
    UPDATE SET target.statistics = source.statistics,
               target.path = source.path
WHEN NOT MATCHED THEN
    INSERT (statistics, path) VALUES (source.statistics, source.path)
               ;
"""

spark.sql(query)

# COMMAND ----------

display(spark.sql(f"SELECT * from {catalog}.{schema}.{gold_table}"))

# COMMAND ----------

def promote_silver_to_gold(
    validated_data: tuple[str, str, dict[int, list[float]]],
    catalog: str,
    schema: str,
    gold_table: str,
    silver_table: str,
) -> None:
    
    """
    Function to promote silver data to gold by updating the gold table
    validated_data: uuids AND image hashes of the images along with the bounding boxes validated by an end user
    catalog: the catalog of the gold table
    schema: the schema of the gold table
    gold_table: the name of the gold table
    silver_table: the name of the silver table
    """

    values = ", ".join([f"('{x}', {y} , '{json.dumps(z)}')" for (x, y, z) in func_input])
    uuids = ", ".join([f"'{x}'" for (x, y, z) in func_input])

    query = "DROP VIEW IF EXISTS gold_updates;"
    spark.sql(query)

    # Create a temp view with the validated data
    query = f"""
            CREATE TEMPORARY VIEW gold_updates AS
            WITH temp_data(uuid, imgHash, bboxs) AS (
            VALUES
                {values}
            )

            SELECT from_json(temp.bboxs, 'bbox array<array<float>>') as bbox, temp.uuid, temp.imgHash, silver.requestId, silver.userId, silver.imagePath
            FROM {catalog}.{schema}.{silver_table} AS silver
            JOIN temp_data AS temp
            ON silver.uuid = temp.uuid
            WHERE silver.path in ({uuids});
            """

    spark.sql(query)

    # merge temp view into gold table, merging on image hash to avoud duplicate images in gold table
    query = f"""
            MERGE INTO {catalog}.{schema}.{gold_table} AS target
            USING gold_updates AS source
            ON (target.imgHash = source.imgHash)
            WHEN MATCHED THEN
                UPDATE SET target.bboxs = source.bboxs,
                        target.uuid = source.uuid,
                        target.imgHash = source.imgHash,
                        target.imgPath = source.imgPath,
                        target.requestId = source.requestId,
                        target.userId = source.userId,
                        target.reviewedTime = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN
                INSERT (bboxs, uuid, imgHash, reviewedTime) VALUES (source.bboxs, source.uuid, source.imgHash, CURRENT_TIMESTAMP()
                        ;
            """

    spark.sql(query)

# COMMAND ----------

# from databricks import sql


# with sql.connect(
#     server_hostname="adb-1881246389460182.2.azuredatabricks.net",
#     http_path="/sql/1.0/warehouses/8605a48953a7f210",
#     access_token="dapi4cd602a58259c1b9cc58c2fbd11bf68b-3",
# ) as connection:
#     with connection.cursor() as cursor:
#         query = f"""
#                 MERGE INTO {catalog}.{schema}.{gold_table} AS target
#                 USING (
#                     SELECT s.*, m.bbox AS new_bbox
#                     FROM {catalog}.{schema}.{silver_table} AS s
#                     JOIN (
#                         VALUES
#                             {mappings}
#                     ) AS m(uuid, bbox) ON s.uuid = m.uuid -- update bbox with the one made by epi's
#                 )
#                 AS source -- the source is the filtered silver table with the new bbox made by the epi's
#                 ON target.sha1 = source.sha1 -- Join condition: merge on image hashes bc no duples allowed in gold table
#                 WHEN MATCHED THEN UPDATE SET
#                             target.modificationTime = source.modificationTime,
#                             target.bbox = source.bbox,
#                             target.confidence = source.confidence

#                 WHEN NOT MATCHED THEN INSERT (id, modificationTime, bbox, confidence)
#                 VALUES (id, modificationTime, bbox, confidence);
#                 """

#         cursor.execute(query)
#         # result = cursor.fetchall()

#         # for row in result:
#         #    print(row)

# COMMAND ----------

# with sql.connect(
#     server_hostname="adb-1881246389460182.2.azuredatabricks.net",
#     http_path="/sql/1.0/warehouses/8605a48953a7f210",
#     access_token="dapi4cd602a58259c1b9cc58c2fbd11bf68b-3",
# ) as connection:
#     with connection.cursor() as cursor:
#         cursor.execute(
#             "SELECT * FROM edav_dev_csels.towerscout_test_schema.test_image_silver LIMIT 2"
#         )
#         result = cursor.fetchall()

#         for row in result:
#             print(row)

# COMMAND ----------

# connection = sql.connect(
#     server_hostname="adb-1881246389460182.2.azuredatabricks.net",
#     http_path="/sql/1.0/warehouses/8605a48953a7f210",
#     access_token="<access-token>",
# )

# cursor = connection.cursor()

# cursor.execute("SELECT * from range(10)")
# print(cursor.fetchall())

# cursor.close()
# connection.close()

# COMMAND ----------


