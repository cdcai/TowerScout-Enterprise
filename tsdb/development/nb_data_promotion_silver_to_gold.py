# Databricks notebook source
# MAGIC %md
# MAGIC 1. Create temp view of rows that have matching uuids from the input
# MAGIC 2. Go through and update the bbox's in the temp view to be the ones from the function input
# MAGIC 3. Do a merge (merging on image hash sha1) into the gold table from the temp view
# MAGIC 4. Delete temp view 

# COMMAND ----------

# %sql
# DELETE FROM edav_dev_csels.towerscout.test_image_gold; -- Delete all records in testing gold table

# COMMAND ----------

# %sql
# DROP TABLE edav_dev_csels.towerscout.test_image_gold; -- Delete testing gold table

# COMMAND ----------

# %sql
# USE CATALOG edav_dev_csels;
# USE SCHEMA towerscout;

# CREATE TABLE  IF NOT EXISTS test_image_gold (
#   user_id STRING,
#   request_id STRING,
#   uuid STRING,
#   reviewed_time TIMESTAMP,
#   bboxes ARRAY<STRUCT<class: INT, x1: FLOAT, y1: FLOAT, x2: FLOAT, y2: FLOAT, conf: FLOAT>>,
#   image_hash INT,
#   image_path STRING,
#   split_label STRING
#   );

# COMMAND ----------

import json
from tsdb.utils.silver_to_gold import create_gold_merge_query, create_updates_view_query
from numpy.random import choice

# COMMAND ----------

catalog = "edav_dev_csels"
schema = "towerscout"
silver_table = "test_image_silver"
gold_table = "test_image_gold"

# COMMAND ----------

validated_data = (
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
            {"conf": 0.8, "class": 0, "x1": 1.2, "x2": 75.4, "y1": 55.1, "y2": 98.2},
            {"conf": 0.9, "class": 0, "x1": 1.2, "x2": 2.3, "y1": 3.4, "y2": 7.5},
        ],
    ),
)

# COMMAND ----------

values = ", ".join(
    [
        f"('{uuid}', '{image_hash}', '{json.dumps(bboxes)}', '{choice(a=['train', 'val', 'test'], p=[0.6, 0.2, 0.2])}')"
        for (uuid, image_hash, bboxes) in validated_data
    ]
)

uuids = ", ".join([f"'{uuid}'" for (uuid, image_hash, bboxes) in validated_data])

# COMMAND ----------

drop_existing_view = "DROP VIEW IF EXISTS gold_updates;"

# COMMAND ----------

spark.sql(drop_existing_view)

# COMMAND ----------

create_updates_view = create_updates_view_query(catalog, schema, silver_table, values, uuids)

# COMMAND ----------

spark.sql(create_updates_view)

# COMMAND ----------

display(spark.sql("SELECT * from gold_updates"))

# COMMAND ----------

df = spark.sql("SELECT * from gold_updates")

# COMMAND ----------

merge_updates_into_gold = create_gold_merge_query(catalog, schema, gold_table)

# COMMAND ----------

spark.sql(merge_updates_into_gold)

# COMMAND ----------

# validated_data = (
#     (
#         "abfss://ddphss-csels@davsynapseanalyticsdev.dfs.core.windows.net/PD/TowerScout/Unstructured/test_images/tmp5j0qh5k121.jpg",
#         "ab3c5de",
#         [ {'label':0, 'xmin':8.2, 'xmax':5.4, 'ymin':5.1, 'ymax':9.2}, {'label':0, 'xmin':0.2, 'xmax':3.3, 'ymin':55.4, 'ymax':3.5}]
#     ),
#     (
#         "abfss://ddphss-csels@davsynapseanalyticsdev.dfs.core.windows.net/PD/TowerScout/Unstructured/test_images/tmp5j0qh5k119.jpg",
#         "g4mf8n",
#         [ {'label':0, 'xmin':1.2, 'xmax':75.4, 'ymin':55.1, 'ymax':98.2}, {'label':0, 'xmin':1.2, 'xmax':2.3, 'ymin':3.4, 'ymax':7.5}]
#     )
# )

# values = ", ".join(
#     [f"('{path}', '{imgHash}', '{json.dumps(annotations)}')" for (path, imgHash, annotations) in validated_data]
# )

# paths = ", ".join([f"'{path}'" for (path, imgHash, annotations) in validated_data])

# delete_existing_records = f"DELETE FROM edav_dev_csels.towerscout_test_schema.test_image_gold WHERE path IN ({paths});"

# spark.sql(delete_existing_records) # delete exisiting record from test gold table for test

# drop_existing_view = "DROP VIEW IF EXISTS gold_updates;"
# spark.sql(drop_existing_view)

# create_updates_view = f"""
#         CREATE TEMPORARY VIEW gold_updates AS
#         WITH temp_data(path, imgHash, annotations) AS (
#         VALUES
#             {values}
#         )

#         SELECT from_json(temp.annotations, 'array<struct<`xmin`:float, `xmax`:float, `ymin`:float, `ymax`:float, `label`:int>>') as annotations, temp.path, temp.imgHash
#         FROM edav_dev_csels.towerscout_test_schema.test_image_silver AS silver
#         JOIN temp_data AS temp
#         ON silver.path = temp.path
#         WHERE silver.path in ({paths});
#         """
# spark.sql(create_updates_view)

# merge_updates_into_gold = f"""
#             MERGE INTO edav_dev_csels.towerscout_test_schema.test_image_gold AS target
#             USING gold_updates AS source
#             ON (target.imgHash = source.imgHash)
#             WHEN MATCHED THEN
#                 UPDATE SET target.annotations = source.annotations,
#                         target.path = source.path,
#                         target.imgHash = source.imgHash
#             WHEN NOT MATCHED THEN
#                 INSERT (annotations, path, imgHash) VALUES (source.annotations, source.path, source.imgHash);
#             """

# spark.sql(merge_updates_into_gold)

# COMMAND ----------

# %sql
# DROP TABLE edav_dev_csels.towerscout_test_schema.test_image_gold;

# COMMAND ----------

# %sql
# CREATE TABLE edav_dev_csels.towerscout_test_schema.test_image_gold (path STRING, imgHash STRING, annotations array<struct<`xmin`:float, `xmax`:float, `ymin`:float, `ymax`:float, `label`:int>>);

# COMMAND ----------

# %sql
# CREATE TABLE edav_dev_csels.towerscout_test_schema.test_image_gold (path STRING, statistics struct<`mean`:array<double>,`median`:array<int>,`stddev`:array<double>,`extrema`:array<array<int>>>);

# COMMAND ----------

# catalog = "edav_dev_csels"
# schema = "towerscout_test_schema"
# gold_table = "test_image_gold"
# silver_table = "test_image_silver"


# func_input = (
#     (
#         "abfss://ddphss-csels@davsynapseanalyticsdev.dfs.core.windows.net/PD/TowerScout/Unstructured/test_images/tmp5j0qh5k121.jpg",
#         [
#             {"ymax": 0.028906, "xmin": 0.169141, "label": 0, "ymin": 0.551953, "xmax": 0.030469},
#             {"ymax": 0.028906, "xmin": 0.183984, "label": 0, "ymin": 0.530859, "xmax": 0.030469},
#         ],

#     ),
#     (
#         "abfss://ddphss-csels@davsynapseanalyticsdev.dfs.core.windows.net/PD/TowerScout/Unstructured/test_images/tmp5j0qh5k115.jpg",
#         [
#             {"ymax": 0.049219, "xmin": 0.441797, "label": 0, "ymin": 0.290234, "xmax": 0.039844},
#             {"ymax": 0.049219, "xmin": 0.461328, "label": 0, "ymin": 0.283984, "xmax": 0.039844},
#         ]
#     ),
# )

# paths = ", ".join([f"'{x}'" for (path, bboxes, img_hash) in func_input])

# COMMAND ----------

# mappings = ", ".join([f"('{x}', '{json.dumps(y)}')" for (path, bboxes, img_hash) in func_input])

# print(mappings)

# COMMAND ----------

# query = "DROP VIEW IF EXISTS gold_updates;"
# spark.sql(query)

# query = f"""
# CREATE TEMPORARY VIEW gold_updates AS
# WITH temp_data(path, statistics) AS (
#   VALUES
#     {mappings}
# )

# SELECT from_json(temp.statistics, 'mean array<double>, median array<int>, stddev array<double>, extrema array<array<int>>') as statistics, temp.path, silver.length
# FROM {catalog}.{schema}.{silver_table} AS silver

# JOIN temp_data AS temp
# ON silver.path = temp.path
# WHERE silver.path in ({paths});
# """

# display(spark.sql(query))

# # uuid's bounding boxes, image hash, userID, set review time as CURRENT_TIMESTAMP()
# # get requestID from silver table

# COMMAND ----------

# display(spark.sql("SELECT * from gold_updates"))

# COMMAND ----------

# display(spark.sql(f"SELECT * from {catalog}.{schema}.{gold_table}"))

# COMMAND ----------

# query = f"""
# MERGE INTO {catalog}.{schema}.{gold_table} AS target
# USING gold_updates AS source
# ON (target.path = source.path)
# WHEN MATCHED THEN
#     UPDATE SET target.statistics = source.statistics,
#                target.path = source.path
# WHEN NOT MATCHED THEN
#     INSERT (statistics, path) VALUES (source.statistics, source.path)
#                ;
# """

# spark.sql(query)

# COMMAND ----------

# display(spark.sql(f"SELECT * from {catalog}.{schema}.{gold_table}"))
