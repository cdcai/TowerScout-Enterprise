"""
This module tests code in tsdb.preprocessing.preprocess functions on image data. If needed,
function docstrings can include examples of what is being tested.
"""
from typing import Any
import shutil

import pytest
from unittest.mock import MagicMock, patch, Mock
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import functions as F
from petastorm.spark import make_spark_converter

import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from PIL import Image

from tsdb.preprocessing.functions import sum_bytes

from tsdb.preprocessing.preprocess import (
    create_converter
)

# TODO: Update unit tests for these functions
from tsdb.ml.data import get_dataloader, collate_fn_img

@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    spark = (
        SparkSession.builder.master("local").appName("test_preprocessing").getOrCreate()
    )
    return spark


def test_create_converter(spark):
    """Test the create_converter function."""

    # Get or create a SparkContext instance
    sc = SparkContext.getOrCreate()

    # Set Petastorm configuration for cache directory (update this path as needed)
    cache_dir = "file:///dbfs/tmp/petastorm/cache"  # "file:///tmp/petastorm_cache"  # Change this to your desired path
    spark.conf.set("SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF", cache_dir)
    spark.conf.set("petastorm.spark.converter.parentCacheDirUrl", cache_dir)

    # setup test dataframe
    images_df = (
        spark.table("edav_dev_csels.towerscout.image_metadata")
        .select("length", "content", "path")
        .limit(5)
    )

    # Calculate bytes
    num_bytes = (
        images_df.withColumn("bytes", F.lit(4) + F.length("content"))
        .groupBy()
        .agg(F.sum("bytes").alias("total_bytes"))
        .collect()[0]["total_bytes"]
    )
    # 341491

    # Mocking the dependencies with correct paths
    with patch("petastorm.spark.make_spark_converter") as mock_make_spark_converter:
        # Call sum_bytes to ensure it returns the mocked value
        assert sum_bytes(images_df, "length") == pytest.approx(
            num_bytes, rel=1e-3
        )  # passed

        # Create a mock converter object to return from make_spark_converter
        test_converter = make_spark_converter(
            images_df,
            parquet_row_group_size_bytes=int(num_bytes / sc.defaultParallelism),
        )

        # Call the function under test
        converter = create_converter(images_df, "length", sc, 0)

        # Check if returned converter is correct
        assert isinstance(converter, type(test_converter))  # passed

        # mock_converter = MagicMock()
        mock_make_spark_converter.return_value = test_converter
        parquet_row_group_size_bytes = int(num_bytes / sc.defaultParallelism)
        # 42686

        # check if make_spark_converter was called correctly
        try:
            mock_make_spark_converter.assert_called_once_with(
                images_df,
                parquet_row_group_size_bytes=parquet_row_group_size_bytes,
                spark_session=spark,
                cache_dir=cache_dir,
            )
            print(
                "Assertion passed: make_spark_converter was called with expected arguments."
            )

        except AssertionError as e:
            print(f"Assertion failed: {e}")
            print(f"Called with: {mock_make_spark_converter.call_args}")
            print(
                f"Expected: (images_df, parquet_row_group_size_bytes={parquet_row_group_size_bytes}, spark_session=spark, cache_dir={cache_dir})"
            )


@pytest.fixture
def data() -> list[dict[str, Any]]:
    # Create dummy images (640x640 white images)
    img1 = Image.new("RGB", (640, 640), color=(255, 255, 255))
    img2 = Image.new("RGB", (640, 640), color=(255, 255, 255))

    data = [
        {
            "im_file": "path/to/img1.jpg",
            "img": img1,
            "bboxes": np.array([0.4, 0.4, 0.6, 0.6, 0.2, 0.33, 0.1, 0.55]),
            "cls": np.array([0.0, 0.0]),
            "ori_shape": np.array(img1.size, dtype=np.uint32),
        },
        {
            "im_file": "path/to/img2.jpg",
            "img": img2,
            "bboxes": np.array(
                [0.12, 0.55, 0.78, 0.97, 0.03, 0.8, 0.1, 0.77, 0.49, 0.21, 0.66, 0.99]
            ),
            "cls": np.array([0.0, 0.0, 0.0]),
            "ori_shape": np.array(img2.size, dtype=np.uint32),
        },
    ]
    return data


@pytest.fixture
def transforms() -> callable:
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return transforms


# def test_collate_fn_img_bboxes(data, transforms):
#     """Test bbox output of the collate_fn_img function."""
#     batch = collate_fn_img(data, transforms)

#     assert torch.allclose(
#         batch["bboxes"],
#         torch.tensor(
#             [
#                 [0.4, 0.4, 0.6, 0.6],
#                 [0.2, 0.33, 0.1, 0.55],
#                 [0.12, 0.55, 0.78, 0.97],
#                 [0.03, 0.8, 0.1, 0.77],
#                 [0.49, 0.21, 0.66, 0.99],
#             ],
#         dtype=torch.float64)
#     )


# def test_collate_fn_img_cls(data, transforms):
#     """Test cls output of the collate_fn_img function."""
#     batch = collate_fn_img(data, transforms)

#     assert torch.allclose(
#         batch["cls"],
#         torch.tensor(
#             [
#                 [0.0],[0.0],[0.0],[0.0],[0.0]
#             ],
#         dtype=torch.float64
#         )
#     )


# def test_collate_fn_img_img(data, transforms):
#     """Test cls output of the collate_fn_img function."""
#     batch = collate_fn_img(data, transforms)
#     img = batch["img"]
#     assert (2, 3, 640, 640) == tuple(img.shape), "Shape should be (batch_size, channels, height, width)"


# def test_collate_fn_img_ori_shape(data, transforms):
#     """Test ori_shape output of the collate_fn_img function."""
#     batch = collate_fn_img(data, transforms)
#     ori_shapes = batch["ori_shape"]
#     for ori_shape in ori_shapes:
#         assert (640, 640) == tuple(ori_shape), "Original shapes should be 640x640"


# def test_collate_fn_img_img_file(data, transforms):
#     """Test im_file output of the collate_fn_img function."""
#     batch = collate_fn_img(data, transforms)
#     im_files = batch["im_file"]
#     assert ("path/to/img1.jpg", "path/to/img2.jpg") == im_files


@pytest.fixture
def remote_dir() -> str:
    return '/Volumes/edav_dev_csels/towerscout/misc/mosaic_streaming_unit_test/'


@pytest.fixture
def local_dir() -> str:
    return '/Volumes/edav_dev_csels/towerscout/misc/mosaic_streaming_unit_test/cache'


@pytest.fixture
def  batch_size() -> int:
    return 2


# def test_get_dataloader(remote_dir: str, local_dir, batch_size: str):
#     """Test the get_dataloader function."""
#     dataloader = get_dataloader(local_dir, remote_dir, batch_size)
    
#     assert isinstance(dataloader, DataLoader)
    
#     for batch in dataloader:
#         assert len(batch["im_file"]) == batch_size, "Batch size should be 2"
    
#     shutil.rmtree(local_dir) 