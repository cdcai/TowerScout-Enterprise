# Databricks notebook source
!pip install databricks-sql-connector

# COMMAND ----------

spark.sql("SELECT * FROM edav_dev_csels.towerscout_test_schema.test_image_silver LIMIT 2").collect()[0]["imageBinary"]

# COMMAND ----------

squares = [(i, i * i) for i in range(4)]
values = ",".join([f"({x}, {y})" for (x, y) in squares])
print(f"{squares} \n {values}")

# COMMAND ----------

from databricks import sql

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
                        server_hostname = "adb-1881246389460182.2.azuredatabricks.net",
                        http_path = "/sql/1.0/warehouses/8605a48953a7f210",
                        access_token = "<access-token>")

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
