from collections import defaultdict
from functools import partial
from typing import Any

import numpy as np
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from tsdb.preprocessing.functions import sum_bytes
from torchvision.transforms import v2
import torchvision
import torch
from torch.utils.data import DataLoader
from PIL import Image
from streaming import StreamingDataset, MDSWriter  # mosiacml-streaming

"""
This module contains higher level preprocessing workflows
that use a combination of tsdb.preprocessing.functions
"""

def create_converter(
    dataframe, bytes_column: "ColumnOrName", sc: SparkContext, parallelism: int = 0
) -> SparkDatasetConverter:
    """
    Returns a PetaStorm converter created from dataframe.

    Args:
        dataframe: DataFrame
        byte_column: Column containing byte count, Used by the petastorm cache
        parallelism: integer for parallelism, used to create petastorm cache
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


def data_augmentation(
    rotation_angle: int = 15,
    prob_H_flip: float = 0.2,
    prob_V_flip: float = 0.2,
    blur: tuple[int, float] = (1, 0.1),
) -> list:
    """
    Data Augmentation function to add label invariant transforms to training pipeline
    Applies a series of transformations such as rotation, horizontal and vertical flips, and Gaussian blur to each image

    TODO: test this
    """
    transforms = [
        v2.RandomRotation(rotation_angle),
        v2.RandomHorizontalFlip(prob_H_flip),
        v2.RandomVerticalFlip(prob_V_flip),
        v2.GaussianBlur(kernel_size=blur[0], sigma=blur[1]),
    ]
    return transforms


# Put these funcitons into preprocess.py not ml_utils!
# Make new branch off this one after you merge main into it AND open a PR.
def convert_to_mds(
    df: DataFrame, columns: dict[str, str], compression: str, out_root: str, **kwargs
) -> None:
    """
    Function that converts a Spark DataFrame to a collection of `.mds` files.
    This is a wrapper for the MDSWriter class:
    https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.MDSWriter.html#mdswriter

    Args:
    df: The spark dataframe to be converted
    columns: A dictionary which contains the column names of the dataframe
            as keys and the corresponding mds data types as values
    compression: Compression algorithm name to use
    out_root: The local or remote directory path to store the output compressed files
    """
    pd_df = df.toPandas()
    samples = pd_df.to_dict("records")

    for sample in samples:
        sample["img"] = Image.open(sample["im_file"])
        sample["ori_shape"] = np.array(sample["img"].size, dtype=np.uint32)

    # Use `MDSWriter` to iterate through the input data and write to a collection of `.mds` files.
    # Note this has been unit tested here: https://github.com/mosaicml/streaming/blob/main/tests/test_writer.py
    with MDSWriter(
        out=out_root, columns=columns, compression=compression, **kwargs
    ) as out:
        for sample in samples:
            out.write(sample)


def collate_fn_img(data: list[dict[str, Any]], transforms: callable) -> dict[str, Any]:
    """
    Function for collating data into batches. Some additional
    processing of the data is done in this function as well
    to get the batch into the format expected by the Ultralytics
    DetectionModel class.

    Args:
        data: The data to be collated a batch
        transforms: Torchvision transforms applied to the PIL images
    Returns: A dictionary containing the collated data in the formated
            expected by the Ultralytics DetectionModel class
    """
    result = defaultdict(list)

    for index, element in enumerate(data):
        for key, value in element.items():
            if key == "img":
                value = transforms(value)

                # height & width after sime transform/augmentation has been done
                _, height, width = value.shape
                result["resized_shape"].append((height, width))
                
                # from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/base.py#L295
                result["ratio_pad"].append(
                    (
                        result["resized_shape"][index][0] / element["ori_shape"][0],
                        result["resized_shape"][index][1] / element["ori_shape"][1],
                    )
                )  

            result[key].append(value)

        num_boxes = len(element["cls"])

        for _ in range(num_boxes):
            result["batch_idx"].append(
                float(index)
            )  # yolo dataloader has this as a float not int

    result["img"] = torch.stack(result["img"], dim=0)
    result["batch_idx"] = torch.tensor(result["batch_idx"])

    # Shape of resulting tensor should be (num_bboxes_in_batch, 4)
    result["bboxes"] = torch.tensor(np.concatenate(result["bboxes"])).reshape(-1, 4)

    # Shape of resulting tensor should be (num_bboxes_in_batch, 1)
    # Call np.concatenate to avoid calling tensor on a list of np arrays but instead just one 2d np array
    result["cls"] = torch.tensor(np.concatenate(result["cls"])).reshape(-1, 1)

    result["im_file"] = tuple(result["im_file"])
    result["ori_shape"] = tuple(result["ori_shape"])
    result["resized_shape"] = tuple(result["resized_shape"])
    result["ratio_pad"] = tuple(result["ratio_pad"])

    return dict(result)


def get_dataloader(
    local_dir: str,
    remote_dir: str,
    batch_size: int,
    transforms: list[callable] = None,
    **kwargs
) -> DataLoader:
    """
    Function that creates a PyTorch DataLoader from a collection of `.mds` files.

    Args:
    local_dir: Local directory where dataset is cached during training
    remote_dir: The local or remote directory where dataset `.mds` files are stored
    batch_size: The batch size of the dataloader and dataset.
          See: https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/faqs_and_tips.html
    transforms: A list of torchvision transforms to be composed and applied to the images
    Returns:
    A PyTorch DataLoader object
    """

    # Note that StreamingDataset is unit tested here: https://github.com/mosaicml/streaming/blob/main/tests/test_streaming.py
    dataset = StreamingDataset(
        local=local_dir,
        remote=remote_dir,
        batch_size=batch_size,
        split=None,
        shuffle=True,
        **kwargs
    )

    if transforms is None:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transform = torchvision.transforms.Compose(transforms)

    # Create PyTorch DataLoader
    collate_fn = partial(collate_fn_img, transforms=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn
    )  # test this to make sure u get a DataLoader u can iterate through
    return dataloader
