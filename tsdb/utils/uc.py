"""
This module contains classes and functions that interact with unity catalog
"""
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Iterable

from pyspark.sql import SparkSession
from pyspark.sql.types import Row
from pyspark.sql.functions import col
from pyspark.dbutils import DBUtils

SchemaInfo = namedtuple("SchemaInfo", ["name", "location"])

@dataclass
class CatalogInfo:
    """
    A class to represent catalog information. You can retrieve schemas from the catalog
    using the catalog name.

    e.g.
    catalog = CatalogInfo.from_spark_config(spark)
    print(catalog[<your_schema_name>])

    Attributes:
        name: The name of the catalog.
        schemas: List containing all volume schemas and their location
    """
    name: str
    schemas: list[SchemaInfo]
    _schema_dict: dict[str, SchemaInfo] = field(init=False)

    def __post_init__(self):
        self._schema_dict = {
            schema.name: schema for schema in self.schemas
        }
    
    def __getitem__(self, key: str) -> SchemaInfo:
        """
        Provides hash based access to catalog schemas

        Args:
            key: The name of the schema to retrieve
        
        Returns:
            SchemaInfo: The schema information
        
        Raises:
            KeyError: If the schema does not exist in the catalog
        """
        return self._schema_dict[key]
    
    def __len__(self) -> int:
        return len(self.schemas)

    def __iter__(self) -> Iterable[SchemaInfo]:
        return iter(self.schemas)

    @classmethod
    def from_spark_config(cls, spark_session: SparkSession, dbutils_: DBUtils=None) -> "CatalogInfo":
        """
        Create a CatalogInfo instance from Spark cluster configuration.

        Args:
            spark (SparkSession): The Spark session object. Note that this must be the 
            Spark session object of the notebook or workflow you are running this code in.

        Returns:
            CatalogInfo: An instance of CatalogInfo with the catalog details.
        """
        # Get the initial catalog name from Spark configuration
        initial_catalog_name = (
            spark_session.conf.get("spark.databricks.sql.initial.catalog.name")
        )

        if not initial_catalog_name:
            panic("Initial catalog name is empty in cluster", dbutils_)
        
        schema_info = cls.query_schema_info(initial_catalog_name, spark_session)
        
        display(schema_info)
        if not schema_info:
            panic("No schema exists in the catalog", dbutils_)
        
        # Set in namedtuple for easy access
        volumes = [
            SchemaInfo(schema["volume_schema"], schema["storage_location"])
            for schema in schema_info
        ]

        # Instantiate
        return cls(initial_catalog_name, volumes)

    def panic(self, error_message: str, dbutils: DBUtils):
        """
        If dbutils is provided, prints the error message and exits the notebook.
        Otherwise, raises a ValueError.

        """
        if dbutils:
            dbutils.notebook.exit(error_message)
            return

        raise ValueError(error_message)

    @staticmethod
    def query_schema_info(initial_catalog_name: str, spark_session: SparkSession) -> list[Row]:
        """
        Returns the EXTERNAL volume schemas and storage locations of the provided catalog
        
        Args:
            initial_catalog_name: Catalog to query
        """
        schema_location = f"{initial_catalog_name}.information_schema.volumes"
        return (
            spark_session
            .table(schema_location)
            .filter(col("volume_type") != "MANAGED")
            .select("volume_schema", "storage_location")
            .collect()
        )
