"""
This module contains DataFrame -> Dataframe transformations that are used in PySpark transform chains
"""
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import tsdb.preprocessing.utils as utils


def compute_bytes(
    dataframe: DataFrame, binary_column: "ColumnOrName", col_name: "str" = "bytes"
) -> DataFrame:
    """
    Returns a dataframe with a bytes column, which calculates the number of
    bytes in a binary column. While this function is intended to be used for
    binary data, we do not strictly enforce this.

    Args:
        dataframe: DataFrame
        binary_column: Name or col that has bit data
        col_name (default="bytes"): The name of the new result column
    """
    base_bytes = 4
    binary_column = utils.cast_to_column(binary_column)
    num_bytes = F.lit(base_bytes) + F.length(binary_column)
    

    return dataframe.withColumn(col_name, num_bytes)
