# Databricks notebook source
#!pip install databricks-sql-connector

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Create temp view of rows that have matching uuids from the input
# MAGIC 2. Go through and update the bbox's in the temp view to be the ones from the function input
# MAGIC 3. Do a merge (merging on image hash sha1) into the gold table from the temp view
# MAGIC 4. Delete temp view 

# COMMAND ----------

#{"mean": [-1,2,-6], "median": [6,6,6], "stddev": [8,8,8], "extrema": [[5,1], [2,21], [6,55]]}

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
        { "mean": [-1.0,2,0,-2.0], "median": [6,6,6], "stddev": [8.0,8.0,8.0], "extrema": [[5,1], [2,21], [6,55]] }
    ),
    (
        "abfss://ddphss-csels@davsynapseanalyticsdev.dfs.core.windows.net/PD/TowerScout/Unstructured/test_images/tmp5j0qh5k115.jpg",
        {"mean": [-1.0,2.0,-2.0], "median": [6,6,6], "stddev": [8.0,8.0,8.0], "extrema": [[5,1], [2,21], [6,55]] }
    ),
)

paths = ", ".join([f"'{x}'" for (x, y) in func_input])

# COMMAND ----------

import json
mappings = ", ".join([f"('{x}', '{json.dumps(y)}')" for (x, y) in func_input])

# mappings = ""
# for (x, y) in func_input:
#     mappings += f"('{x}', 'STRUCT("
#     mappings += f"ARRAY{y['mean']}, ARRAY{y['median']}, ARRAY{y['stddev']}, ARRAY{y['extrema']})'), "

# mappings = mappings[:-2] 

print(mappings)

# COMMAND ----------

# SELECT CAST(new_length AS BIGINT), temp.path

# SELECT CAST(new_statistics AS struct<`mean`:array<double>,`median`:array<int>,`stddev`:array<double>,`extrema`:array<array<int>>>), temp.path

# COMMAND ----------

query = "DROP VIEW IF EXISTS gold_updates;"
spark.sql(query)

query = f"""
CREATE TEMPORARY VIEW gold_updates AS
WITH temp_data(path, statistics) AS (
  VALUES
    {mappings}
)

--SELECT temp.statistics, temp.path
SELECT from_json(temp.statistics, 'mean array<double>, median array<int>, stddev array<double>, extrema array<array<int>>') as statistics, temp.path
FROM silver_temp AS silver
JOIN temp_data AS temp
ON silver.path = temp.path
WHERE silver.path in ({paths});
"""

display(spark.sql(query))

# uuid's bounding boxes, image hash, userID, set review time as CURRENT_TIMESTAMP()
# get request ID from silver table, 

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

# from pyspark.sql.types import (
#     StructField,
#     StructType,
#     FloatType,
#     TimestampType,
#     StringType,
#     IntegerType,
# )

# from databricks import sql

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

# df_silver = spark.read.table("edav_dev_csels.towerscout_test_schema.test_image_silver")
# display(df_silver)

# COMMAND ----------

# def promote_silver_to_gold(
#     silver_csv: str, catalog: str, schema: str, gold_table: str
# ) -> None:
#     """
#     Function to promote silver data to gold by updating the gold table
#     silver_csv: path to the csv of the silver data to be promoted/added to the gold table
#     catalog: the catalog of the gold table
#     schema: the schema of the gold table
#     gold_table: the name of the gold table
#     """

#     df = spark.read.csv(silver_csv, header=True, inferSchema=True)
#     df.reateOrReplaceTempView('temp_delta_table')
#     # dont use spark.slq use the code from bottom of: https://docs.delta.io/latest/delta-update.html#-delta-merge
#     spark.sql(
#         f"""
#     MERGE INTO {catalog}.{schema}.{gold_table} AS target
#     USING temp_delta_table AS source
#     ON target.id = source.id -- Specify the join condition for matching records, merge on image hashes potentially
#     WHEN MATCHED THEN UPDATE SET target.modificationTime = source.modificationTime,
#                 target.bbox = source.bbox,
#                 target.confidence = source.confidence

#     WHEN NOT MATCHED THEN INSERT (id, modificationTime, bbox, confidence)
#     VALUES (id, modificationTime, bbox, confidence);
#     """
#     )
