"""
This module contains low-level functions that transform column objects to a column object. These functions are utility-like but specific to preprocessing tasks.
"""
from typing import Union
import pyspark.sql.functions as F


def sum_column(dataframe, column: "str") -> Union[int, float]:
    """
    Returns the sum of a column in a dataframe.

    Args:
        dataframe: DataFrame
        column: Numeric column or column name containing numeric values.
    """
    result = dataframe.select(F.sum(column)).first()
    return result[0]