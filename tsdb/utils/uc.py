"""
This module contains classes and functions that interact with unity catalog

"""
from collections import namedtuple
from dataclasses import dataclass

from pyspark.sql import SparkSession
from pyspark.sql.types import Row

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