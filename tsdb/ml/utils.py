from collections import namedtuple
from dataclasses import dataclass

from pyspark.sql import SparkSession, DataFrame, Column
from pyspark.sql.types import Row
import pyspark.sql.functions as F

import logging
from logging.handlers import RotatingFileHandler
from logging import Logger

from pathlib import Path

from petastorm.spark import SparkDatasetConverter, make_spark_converter




def create_converter(
    dataframe, bytes_column: "ColumnOrName", parallelism: int = 0
) -> SparkDatasetConverter:
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
        dataframe, parquet_row_group_size_bytes=int(num_bytes / parallelism)
    )

    return converter



def setup_logger(log_path: str, logger_name: str) -> tuple[Logger, RotatingFileHandler]:
    """
    Creates and returns a Logger object

    Args:
        log_path: Path to store the log file
    Returns:
        The Logger object and the RotatingFileHandler object
    """
    # TODO: Log file may become too large, may need to be partitioned by date/week/month

    # Create the log directory if it doesn't exist
    log_dir = str(Path(log_path).parent)

    # Set up logging
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    try:
        # Create a rotating file handler
        handler = RotatingFileHandler(log_path, maxBytes=1000000, backupCount=1)
        handler.setLevel(logging.INFO)

        # Create a logging format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger and then you can use the logger
        logger.addHandler(handler)
    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise e

    return logger, handler




SchemaInfo = namedtuple("SchemaInfo", ["name", "location"])


def cast_to_column(column: "ColumnOrName") -> Column:
    """
    Returns a column data type. Used so functions can flexibly accept
    column or string names.
    """
    if isinstance(column, str):
        column = F.col(column)

    return column


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

    return dataframe.withColumn("bytes", num_bytes)


def sum_bytes(dataframe, bytes_column: "ColumnOrName") -> int:
    """
    Returns the sum of bytes in a bytes column from a dataframe. Primarily used
    to start a Petastorm cache.

    Args:
        dataframe: DataFrame
        bytes_column: Column that contains counts of bytes
    """
    aggregate_bytes = dataframe.agg(F.sum(bytes_column).alias("total_bytes"))
    return aggregate_bytes.collect()[0]["total_bytes"]


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
        initial_catalog_name = spark.conf.get(
            "spark.databricks.sql.initial.catalog.name"
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