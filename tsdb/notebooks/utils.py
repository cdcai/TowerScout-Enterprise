# Databricks notebook source
from collections import namedtuple
from dataclasses import dataclass

from pyspark.sql import SparkSession, DataFrame, Column
from pyspark.sql.types import Row
import pyspark.sql.functions as F

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
    if parallelism == 0:
        parallelism = sc.defaultParallelism

    num_bytes = sum_bytes(dataframe, bytes_column)

    # Cache
    converter = make_spark_converter(
        dataframe, 
        parquet_row_group_size_bytes=int(num_bytes/parallelism)
    )

    return converter
