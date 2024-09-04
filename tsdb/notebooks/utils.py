# Databricks notebook source
from collections import namedtuple
from dataclasses import dataclass

from pyspark.sql import SparkSession, DataFrame, Column
from pyspark.sql.types import Row
import pyspark.sql.functions as F

import json

# COMMAND ----------

SchemaInfo = namedtuple("SchemaInfo", ["name", "location"])


@dataclass
class CatalogInfo:
    """
    A class to represent catalog information.

    Attributes:
        name: The name of the catalog.
        volume: List containing all volume schemas and their location
    """
    name: str
    schemas: list[SchemaInfo]

    @classmethod
    def from_spark_config(cls, spark: SparkSession) -> "CatalogInfo":
        """
        Create a CatalogInfo instance from Spark configuration.

        Args:
            spark (SparkSession): The Spark session object.

        Returns:
            CatalogInfo: An instance of CatalogInfo with the catalog details.
        """
        # Get the initial catalog name from Spark configuration
        initial_catalog_name = (
            spark.conf.get("spark.databricks.sql.initial.catalog.name")
        )

        if not initial_catalog_name:
            dbutils.notebook.exit("Initial catalog name is empty in cluster")
        
        schema_info = cls.query_schema_info(initial_catalog_name)
        
        if not schema_info:
            dbutils.notebook.exit("No schema exists in the catalog")
        
        # Set in namedtuple for easy access
        volumes = [
            SchemaInfo(schema["volume_schema"], schema["storage_location"])
            for schema in schema_info
        ]

        # Instantiate
        return cls(initial_catalog_name, volumes)

    @staticmethod
    def query_schema_info(initial_catalog_name: str) -> list[Row]:
        """
        Returns the volume and storage locations of the provided catalog

        Args:
            initial_catalog_name: Catalog to query
        """
        schema_location = f"{initial_catalog_name}.information_schema.volumes"
        schema_info = spark.sql(
            f"SELECT volume_schema, storage_location FROM {schema_location}"
        )

        return schema_info.collect()

# COMMAND ----------

def cast_to_column(column: "ColumnOrName") -> Column:
    """
    Returns a column data type. Used so functions can flexibly accept
    column or string names.
    """
    if isinstance(column, str):
        column = F.col(column)

    return column

# COMMAND ----------

def compute_bytes(dataframe: DataFrame, binary_column: "ColumnOrName") -> DataFrame:
    """
    Returns a dataframe with a bytes column, which calculates the number of bytes
    in a binary column

    Args:
        dataframe: DataFrame
        binary_column: Name or col that has bit data
    """
    binary_column = cast_to_column(binary_column)
    num_bytes = F.lit(4) + F.length(binary_column)

    return (
        dataframe
        .withColumn("bytes", num_bytes)
    )

# COMMAND ----------

def sum_bytes(dataframe, bytes_column: "ColumnOrName") -> int:
    """
    Returns the sum of bytes in a bytes column from a dataframe. Primarily used
    to start a Petastorm cache.

    Args:
        dataframe: DataFrame
        bytes_column: Column that contains counts of bytes
    """
    aggregate_bytes = dataframe.agg(
        F.sum(bytes_column).alias("total_bytes")
    )
    return aggregate_bytes.collect()[0]["total_bytes"]


# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter

def create_converter(dataframe, bytes_column: "ColumnOrName", parallelism: int=0) -> SparkDatasetConverter:
    """
    Returns a PetaStorm converter created from dataframe.

    Args:
        dataframe: DataFrame
        byte_column: Column that contains the byte count. Used to create the petastorm  cache
        parallelism: integer for parallelism, used to create the petastorm cache
    """
    # Note this uses spark context 
    if parallelism == 0:
        parallelism = sc.defaultParallelism

    num_bytes = sum_bytes(dataframe, bytes_column)

    # Cache
    converter = make_spark_converter(
        dataframe, 
        parquet_row_group_size_bytes=int(num_bytes/parallelism)
    )

    return converter

# COMMAND ----------

def get_cluster_tag_val(key_name: str) -> str:
    # Fetch all cluster tags from the Spark configuration
    all_tags = spark.conf.get("spark.databricks.clusterUsageTags.clusterAllTags", None)

    if all_tags:
        # Parse the JSON-like string into a list of dictionaries
        tags_list = json.loads(all_tags)
        
        # Convert the list of dictionaries into a dictionary for easier access
        tags_dict = {tag['key']: tag['value'] for tag in tags_list}
        
        # Get the value of the specified tag
        tag_value = tags_dict.get(key_name)
        
        if tag_value:
            # Return the tag value if it exists
            return tag_value
        else:
            # Exit the notebook with a failure message if the tag does not exist or has no value
            dbutils.notebook.exit(f"Failure: The tag '{key_name}' does not exist or has no value.")
    else:
        # Display a message if no tags are found or the cluster does not have any tags
        display("No tags found or the cluster does not have any tags.")
        dbutils.notebook.exit(f"Failure: The tag '{key_name}' does not exist or has no value.")

# This function retrieves the catalog name, schema name, and volumes using the filtered catalog and schema names
def get_catalog_schema_config_sql(cat, schema):
    # Query to get the catalog name and schema name from the information schema
    conf = spark.sql(f"SELECT catalog_name, schema_name FROM information_schema.schemata WHERE catalog_name LIKE '%{cat}%' and schema_name LIKE '%{schema}%'")
    # Extract the catalog name and schema name from the query result
    catalog_name = conf.collect()[0]['catalog_name']
    schema_name = conf.collect()[0]['schema_name']
    
    # Query to get the volumes in the specified catalog and schema
    vols = spark.sql(f"show volumes in {catalog_name}.{schema_name}")
    
    # Extract the volume names from the query result
    volumes = [row['volume_name'] for row in vols.select('volume_name').collect()]
    return catalog_name, schema_name, volumes


env = get_cluster_tag_val("edav_environment")
edav_center = get_cluster_tag_val("edav_center").lower() # must lowercase or else LIKE statement will not work
edav_project = get_cluster_tag_val("edav_project").lower()
cluster_catalog_name, cluster_schema_name, volumes = get_catalog_schema_config_sql(edav_center, edav_project)

# Retrieve the config file path from cluster tags
cluster_config_path = get_cluster_tag_val("tower_scout_config_file_name").strip() # for some reason there is a leading space

# Set the retrieved environment, catalog name, and schema name as Spark configuration variables
spark.conf.set("env", env)
spark.conf.set("catalog_name", cluster_catalog_name)
spark.conf.set("schema_name", cluster_schema_name)
spark.conf.set("config_path", cluster_config_path)

if 'configs' in volumes:
    # Construct the volume location path for the config file
    vol_loc = f"/Volumes/{spark.conf.get('catalog_name')}/{spark.conf.get('schema_name')}/configs/{spark.conf.get('config_path')}"
    try:
        # Load the configuration data from the specified volume location
        conf = spark.read.format("json").load(vol_loc, multiLine=True)
        tenant_id = conf.collect()[0]['tenant_id']
        # Set the debug mode from the loaded configuration data based on the environment
        if env == 'development':
            debug_mode = spark.conf.set("debug_mode", conf.collect()[0][env]['debug_mode'])
            unit_test_mode = spark.conf.set("unit_test_mode", conf.collect()[0][env]['unit_test_mode'])
            key_vault_name = spark.conf.set("key_vault_name", conf.collect()[0][env]['key_vault_name'])
            client_id = conf.collect()[0][env]['azure_config']['client_id']

        elif env == 'production':
            debug_mode = spark.conf.set("debug_mode", conf.collect()[0][env]['debug_mode'])
            unit_test_mode = spark.conf.set("unit_test_mode", conf.collect()[0][env]['unit_test_mode'])
            key_vault_name = spark.conf.set("key_vault_name", conf.collect()[0][env]['key_vault_name'])
            client_id = conf.collect()[0][env]['azure_config']['client_id']
        else:
            dbutils.notebook.exit(f"Failure: Enviornment is neither development or production is {env}.")
    except Exception as e:
        # Exit the notebook with an error message if unable to load the configuration data
        dbutils.notebook.exit(f"Failure: Unable to load configuration data, check config file. Error: {str(e)}")
else:
    # Exit the notebook with an error message if the 'configs' volume does not exist
    dbutils.notebook.exit("No config folder in volume")

spark.conf.set("vol_location_configs", 
                   f"/Volumes/{cluster_catalog_name}/{spark.conf.get('schema_name')}/configs/")

# COMMAND ----------

config_data = [
    (
        spark.conf.get('env'), 
        spark.conf.get('catalog_name'), 
        spark.conf.get('schema_name'), 
        spark.conf.get('debug_mode'), 
        spark.conf.get('vol_location_configs'), 
        spark.conf.get('key_vault_name'),
        client_id, 
        tenant_id,
        spark.conf.get('unit_test_mode')
    )
]

config_columns = [
    "env", 
    "catalog_name", 
    "schema_name", 
    "debug_mode", 
    "vol_location_configs",
    "key_vault_name", 
    "client_id", 
    "tenant_id", 
    "unit_test_mode",
]

# Create a DataFrame with the configuration values and column names
config_df = spark.createDataFrame(config_data, config_columns)

# Create a global temporary view for the configuration DataFrame
config_df.createOrReplaceGlobalTempView("global_temp_towerscout_configs")

display(spark.sql("SELECT * FROM global_temp.global_temp_towerscout_configs"))
