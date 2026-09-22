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
