# Databricks notebook source
import json

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
