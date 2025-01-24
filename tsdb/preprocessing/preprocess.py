"""
This module contains higher level preprocessing workflows
that use a combination of tsdb.preprocessing.functions
"""
import numpy as np
from PIL import Image
from pyspark.sql import DataFrame, SparkSession

from streaming import MDSWriter  # mosiacml-streaming


def convert_to_mds(
    df: DataFrame,
    out_root: str,
    columns: dict[str, str] = None,
    compression: str = "zstd",
    **kwargs
) -> None:
    """
    Function that converts a Spark DataFrame to a collection of `.mds` files.
    This is a wrapper for the MDSWriter class:
    https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.MDSWriter.html#mdswriter

    Args:
    df: The spark dataframe to be converted
    out_root: The local or remote directory path to store the output compressed files
    columns: A dictionary which contains the column names of the dataframe
            as keys and the corresponding mds data types as values
    compression: Compression algorithm name to use

    TODO: this implementation is not completely tested
    TODO: refactor to be more general
    """
    if columns is None:
        columns = {
            "image_path": "str",
            "img": "ndarray:uint8:640,640,3",
            "bboxes": "ndarray:float32",
            "cls": "ndarray:float32",
            "ori_shape": "ndarray:uint32",
            "resized_shape": "ndarray:uint32",
        }

    pd_df = df.toPandas()
    samples = pd_df.to_dict("records")

    for sample in samples:
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except:
            print(f"Error reading image {sample['image_path']}. Skipping row.")
            sample["img"] = None
            continue

        sample["ori_shape"] = np.array(img.size, dtype=np.uint32)
        sample["img"] = np.array(img.resize((640, 640)))  # hardcode 640 for now
        sample["resized_shape"] = np.array(sample["img"].shape[:2], dtype=np.uint32)

    # Use `MDSWriter` to iterate through the input data and write to a collection of `.mds` files.
    # Note this has been unit tested here: https://github.com/mosaicml/streaming/blob/main/tests/test_writer.py
    with MDSWriter(
        out=out_root, columns=columns, compression=compression, **kwargs
    ) as out:
        for sample in samples:
            # skip rows whose images are corrupted/caused an error.
            if sample["img"] is not None:
                try:
                    out.write(sample)
                except Exception as e:
                    print(f"Row: {sample} \n caused exception: {e}")


def build_mds_by_splits(catalog: str, schema: str, table: str, out_root_base: str) -> int:
    """
    Given a catalog, schema, table_name for a delta table, creates a MDS dataset for training, validation, and testing.
    Obtains the latest delta table version and persists this information in the path for reproducibility.

    Args:
        catalog: UC name
        schema: UC schema name
        table: UC table name
        out_root_base: Volume path to save data to
    
    TODO: Refactor to handle case where directory is non-empty among other things
    """
    spark = SparkSession.builder.getOrCreate()

    training_data_table = f"{catalog}.{schema}.{table}"
    
    # We get the latest version number for reproducibility
    version_number = (
        spark
        .sql(f"DESCRIBE HISTORY {training_data_table} LIMIT 1")
        .collect()[0]["version"]
    )

    # Read the dataframe, we flatten bboxes to match a mds data type
    # image_path will be read by convert_to_mds
    # cls is extracted from bboxes to match Ultralytics training setup
    dataframe = (
        spark
        .read
        .format("delta")
        .option("versionAsOf", version_number)
        .table(training_data_table)
        .selectExpr(
            "image_path",
            "transform(bboxes, x -> float(0)) AS cls",
            "flatten(transform(bboxes, x -> array(x.x1, x.y1, x.x2, x.y2))) AS bboxes",
            "split_label"
        )
    )

    save_path = f"{out_root_base}/{table}/version={version_number}/"
    
    for split in ("train", "val", "test"):
        split_df = dataframe.filter(f"split_label == '{split}'").drop("split_label")
        convert_to_mds(split_df, out_root=f"{save_path}/{split}")
    
    return version_number
