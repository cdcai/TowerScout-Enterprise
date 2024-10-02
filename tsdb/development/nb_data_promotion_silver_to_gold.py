# Databricks notebook source
!pip install databricks-sql-connector

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Create temp view of rows that have matching uuids from the input
# MAGIC 2. Go through and update the bbox's in the temp view to be the ones from the function input
# MAGIC 3. Do a merge (merging on image hash sha1) into the gold table from the temp view
# MAGIC 4. Delete temp view 

# COMMAND ----------

catalog = "edav_dev_csels"
schema = "towerscout_test_schema"
gold_table = "gold"
silver_table = "test_image_silver"


func_input = (
    (
        "abfss://ddphss-csels@davsynapseanalyticsdev.dfs.core.windows.net/PD/TowerScout/Unstructured/test_images/tmp5j0qh5k121.jpg",
        3,
    ),
    (
        "abfss://ddphss-csels@davsynapseanalyticsdev.dfs.core.windows.net/PD/TowerScout/Unstructured/test_images/tmp5j0qh5k115.jpg",
        5,
    ),
)

paths = ", ".join([f"'{x}'" for (x, y) in func_input])

query = "DROP VIEW IF EXISTS silver_filtered;"
spark.sql(query)


# create temp view, get rows by path (will be uuid in practice)
query = f"""
CREATE TEMPORARY VIEW silver_filtered AS
SELECT *
FROM {catalog}.{schema}.{silver_table}
WHERE path IN ({paths});
"""
spark.sql(query)

print("Paths:", paths)

query = "SELECT * FROM silver_filtered;"
display(spark.sql(query))

# COMMAND ----------

mappings = ", ".join([f"('{x}', '{y}')" for (x, y) in func_input])
print(mappings)

# COMMAND ----------

query = f"""UPDATE silver_filtered
SET length = (
    SELECT new_length
    FROM (
        VALUES 
            {mappings}
    ) AS mapping(path, new_length)
    WHERE mapping.path = silver_filtered.path
);
"""

print(query)

spark.sql(query)

query = "SELECT * FROM silver_filtered;"
display(spark.sql(query))

# COMMAND ----------

# convert to mapping for join query when filtering silver table results

# print(mappings)


# display(
#     spark.sql(
#         f"""SELECT m.path, m.length
#         FROM {catalog}.{schema}.{silver_table} AS s
#         JOIN (
#             VALUES 
#                 {mappings}
#         ) AS m(path, length) ON s.path = m.path;
#     """
#     )
# )  # .collect()[0]["imageBinary"]

# COMMAND ----------

# input will be a list of tuples of the form (uuid, list[bboxs])
func_input = (
    ("21141ff23", [[2.2, 3.1, 4.2, 1.2], [4.2, 7.1, 7.6, 1.1]]),
    ("fhk5343", [[2.2, 3.1, 4.2, 1.2], [4.2, 7.1, 7.6, 1.1]]),
)

# convert to mapping for join query when filtering silver table results
mappings = ", ".join([f"('{x}', '{y}')" for (x, y) in func_input])

print(mappings)

# COMMAND ----------

from databricks import sql


with sql.connect(
    server_hostname="adb-1881246389460182.2.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/8605a48953a7f210",
    access_token="dapi4cd602a58259c1b9cc58c2fbd11bf68b-3",
) as connection:
    with connection.cursor() as cursor:
        query = f"""
                MERGE INTO {catalog}.{schema}.{gold_table} AS target
                USING (
                    SELECT s.*, m.bbox AS new_bbox
                    FROM {catalog}.{schema}.{silver_table} AS s
                    JOIN (
                        VALUES 
                            {mappings}
                    ) AS m(uuid, bbox) ON s.uuid = m.uuid -- update bbox with the one made by epi's 
                ) 
                AS source -- the source is the filtered silver table with the new bbox made by the epi's
                ON target.sha1 = source.sha1 -- Join condition: merge on image hashes bc no duples allowed in gold table
                WHEN MATCHED THEN UPDATE SET 
                            target.modificationTime = source.modificationTime,
                            target.bbox = source.bbox,
                            target.confidence = source.confidence

                WHEN NOT MATCHED THEN INSERT (id, modificationTime, bbox, confidence)
                VALUES (id, modificationTime, bbox, confidence);
                """

        cursor.execute(query)
        # result = cursor.fetchall()

        # for row in result:
        #    print(row)

# COMMAND ----------

with sql.connect(
    server_hostname="adb-1881246389460182.2.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/8605a48953a7f210",
    access_token="dapi4cd602a58259c1b9cc58c2fbd11bf68b-3",
) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM edav_dev_csels.towerscout_test_schema.test_image_silver LIMIT 2"
        )
        result = cursor.fetchall()

        for row in result:
            print(row)

# COMMAND ----------

from pyspark.sql.types import (
    StructField,
    StructType,
    FloatType,
    TimestampType,
    StringType,
    IntegerType,
)

from databricks import sql

# COMMAND ----------

connection = sql.connect(
    server_hostname="adb-1881246389460182.2.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/8605a48953a7f210",
    access_token="<access-token>",
)

cursor = connection.cursor()

cursor.execute("SELECT * from range(10)")
print(cursor.fetchall())

cursor.close()
connection.close()

# COMMAND ----------

df_silver = spark.read.table("edav_dev_csels.towerscout_test_schema.test_image_silver")
display(df_silver)

# COMMAND ----------

schema = StructType([
    StructField("path", StringType()),
    StructField("modificationTime", TimestampType()),
    StructField("bbox", StructType([
        StructField("x1", FloatType()),
        StructField("x2", FloatType()),
        StructField("y1", FloatType()),
        StructField("y2", FloatType()),
        StructField("conf", FloatType()),
        StructField("class", IntegerType())
        StructField("class_name", StringType())
    ]))
])

# Create an empty dataframe with the specified schema
df_gold = spark.createDataFrame([], schema)
display(df_gold)

# COMMAND ----------

def promote_silver_to_gold(
    silver_csv: str, catalog: str, schema: str, gold_table: str
) -> None:
    """
    Function to promote silver data to gold by updating the gold table
    silver_csv: path to the csv of the silver data to be promoted/added to the gold table
    catalog: the catalog of the gold table
    schema: the schema of the gold table
    gold_table: the name of the gold table
    """

    df = spark.read.csv(silver_csv, header=True, inferSchema=True)
    df.reateOrReplaceTempView('temp_delta_table')
    # dont use spark.slq use the code from bottom of: https://docs.delta.io/latest/delta-update.html#-delta-merge
    spark.sql(
        f"""
    MERGE INTO {catalog}.{schema}.{gold_table} AS target
    USING temp_delta_table AS source
    ON target.id = source.id -- Specify the join condition for matching records, merge on image hashes potentially
    WHEN MATCHED THEN UPDATE SET target.modificationTime = source.modificationTime,
                target.bbox = source.bbox,
                target.confidence = source.confidence

    WHEN NOT MATCHED THEN INSERT (id, modificationTime, bbox, confidence)
    VALUES (id, modificationTime, bbox, confidence);
    """
    )
