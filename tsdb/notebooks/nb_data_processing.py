# Databricks notebook source
def split_data(images: DataFrame) -> (DataFrame, DataFrame, DataFrame):
    """
    Splits a Spark dataframe into train, test, and validation sets.

    Args:
        df (DataFrame): Input dataframe to be split.

    Returns:
        tuple: A tuple containing the train, test, and validation dataframes.
    """
    # split the dataframe into 3 sets
    images_train = images.sampleBy(("label"), fractions={0: 0.8, 1: 0.8})
    images_remaining = images.join(
        images_train, on="path", how="leftanti"
    )  # remaining from images
    images_val = images_remaining.sampleBy(
        ("label"), fractions={0: 0.5, 1: 0.5}
    )  # 50% of images_remaining
    images_test = images_remaining.join(
        images_val, on="path", how="leftanti"
    )  # remaining 50% from the images_remaining
    
    return images_train, images_test, images_val


def split_datanolabel(images: DataFrame) -> (DataFrame, DataFrame, DataFrame):
    """
    Splits a Spark dataframe into train, test, and validation sets.

    Args:
        df (DataFrame): Input dataframe to be split.

    Returns:
        tuple: A tuple containing the train, test, and validation dataframes.
    """
    # split the dataframe into 3 sets
    images_train = images.sample(fraction=0.8)
    images_remaining = images.join(
        images_train, on="path", how="leftanti"
    )  # remaining from images
    images_val = images_remaining.sample(fraction=0.5)  # 50% of images_remaining
    images_test = images_remaining.join(
        images_val, on="path", how="leftanti"
    )  # remaining 50% from the images_remaining
    return images_train, images_test, images_val

# COMMAND ----------

def get_converter_df(dataframe: DataFrame) -> callable:
    """
    Creates a petastrom converter for a Spark dataframe

    Args:
        dataframe: The Spark dataframe
    Returns:
        callable: A petastorm converter 
    """
    
    dataframe = dataframe.transform(compute_bytes, "content")
    converter = create_converter(
        dataframe,
        "bytes"
    )
 
    return converter
