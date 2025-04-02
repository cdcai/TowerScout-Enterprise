"""
This module contains utilities for drift detection in the TowerScout application
"""

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def get_struct_counts(df: DataFrame, timestamp_col: str, array_struct_col: str, filter_clause: str, time_window_days: int = 90) -> DataFrame:
    """
    This function takes in a dataframe which contains a timestamp column and a column that contains arrays of structs 
    and returns a dataframe with a column that contains the number of structs in the array that match the 
    filter clause given for that row as well as the total number of structs in the array for that row. This is made generic put it's
    primary purpose is to be used to compute the number of bounding boxes which were deselected by end users. 

    Args:
        df: The input dataframe
        timestamp_col: The name of the timestamp column in the input dataframe
        array_struct_col: The name of the column that contains arrays of structs in the input dataframe
        filter_clause: A SQL expression that filters the structs in the array_struct_col column
        time_window_days: The number of days to consider for the computation of the number of structs in the array
    Returns:
        A dataframe with columns `num_filtered_structs` which contains the number of structs in the array 
        that match the filter clause and `num_structs` which contains the total number of structs in the array for that row
    """
    
    df_with_counts = df.filter(
        F.expr(f"{timestamp_col} >= date_sub(current_date(), {time_window_days})")  # retrieve records from last `time_window_days` days
    ).select("*",
        F.size(F.expr(f"filter({array_struct_col}, x -> {filter_clause})")).alias("num_filtered_structs"),  # compute number of deselected bboxes per row
        F.size(array_struct_col).alias("num_structs")  # compute total number of detected bboxes per row
    )

    return df_with_counts


def compute_drift_score(df: DataFrame, num_filtered_structs_col: str, num_structs_col: str) -> float:
    """
    This function computes the drift score for a given dataframe. The drift score is defined as the ratio of the number of structs (bounding boxes) that were deselected (filtered) by end users to the total number of structs in the array.

    Args:
        df: The input dataframe
        num_filtered_structs_col: The name of the column that contains the number of structs that were deselected (filtered) by end users
        num_structs_col: The name of the column that contains the total number of structs in the array
    Returns:
        The drift score (ratio) as a float
    """

    aggregated_num_structs = df.agg({num_structs_col: "sum"})
    aggregated_num_filtered_structs = df.agg({num_filtered_structs_col: "sum"})

    total_structs = aggregated_num_structs.collect()[0][0]
    total_filtered_structs = aggregated_num_filtered_structs.collect()[0][0]

    return total_filtered_structs / total_structs