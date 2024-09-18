"""
This module contains utility functions that don't directly do any processing, but help with processing.
"""
from pyspark.sql import Column
import pyspark.sql.functions as F


def cast_to_column(column: "ColumnOrName") -> Column:
    """
    Returns a column data type. Used so functions can flexibly accept
    column or string names.
    """
    if isinstance(column, str):
        column = F.col(column)

    return column