import shutil
from typing import Any
import uuid

import pytest
import torch
import numpy as np
from pyspark.sql import SparkSession

import ultralytics.utils as uutils

from tsdb.ml.types import Hyperparameters, ImageMetadata
from tsdb.ml.datasets import (
    get_dataloader,
    collate_fn_img,
    DataLoader,
    DataLoaders,
    YoloDataset,
    ImageBinaryDataset
)


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """
    Returns the SparkSession to be used in tests that require Spark
    """
    spark = (
        SparkSession.builder.master("local")
        .appName("test_data_processing")
        .getOrCreate()
    )
    
    return spark


@pytest.fixture
def catalog(spark: SparkSession) -> str:
    if spark.catalog._jcatalog.tableExists("global_temp.global_temp_towerscout_configs"):
        configs = spark.sql("SELECT * FROM global_temp.global_temp_towerscout_configs").collect()[0]
        catalog = configs["catalog_name"]

    else:
        RaiseException("Global view 'global_temp_towerscout_configs' does not exist, make sure to run the utils notebook")
    
    return catalog


@pytest.fixture
def schema(spark: SparkSession) -> str:
    if spark.catalog._jcatalog.tableExists("global_temp.global_temp_towerscout_configs"):
        configs = spark.sql("SELECT * FROM global_temp.global_temp_towerscout_configs").collect()[0]
        schema = configs["schema_name"]

    else:
        RaiseException("Global view 'global_temp_towerscout_configs' does not exist, make sure to run the utils notebook")
    
    return schema


@pytest.fixture
def image_binary_dir(catalog: str, schema: str) -> str:
    return f"/Volumes/{catalog}/{schema}/misc/unit_tests/image_binary_dataset/"


@pytest.fixture
def remote_dir(catalog: str, schema: str) -> str:
    return f"/Volumes/{catalog}/{schema}/misc/unit_tests/mosaic_streaming_unit_test/"


@pytest.fixture
def local_dir() -> str:
    # append uuid to avoid using same cache location between tests
    return "/local/cache/path/" + str(uuid.uuid4())


@pytest.fixture
def hyperparams() -> Hyperparameters:
    hyperparams = Hyperparameters(
        lr0=0.001,
        momentum=0.9,
        weight_decay=0.005,
        batch_size=2,
        epochs=3,
        prob_H_flip=0.5,
        prob_V_flip=0.5,
        prob_mosaic=0.5,
    )
    return hyperparams


def test_image_binary_dataset(spark: SparkSession, image_binary_dir: str):
    image_df = (
    spark
    .read
    .format("binaryFile")
    .load(image_binary_dir)
    .select("content")
    .limit(10)
    )

    image_df = image_df.toPandas()
    image_bins = image_df["content"]

    bin_dataset = ImageBinaryDataset(image_bins)

    assert len(bin_dataset) == len(image_df), f"Dataset length should be the same as dataframe length which is {len(image_df)}"
    assert isinstance(bin_dataset, ImageBinaryDataset), "Dataset should be of type ImageBinaryDataset"
    assert isinstance(bin_dataset[0], dict), "Dataset item should be of type dict"


def test_get_image_and_label(remote_dir: str, local_dir, hyperparams: Hyperparameters):
    dataset = YoloDataset(
        remote=remote_dir,
        local=local_dir,
        shuffle=True,
        hyperparameters=hyperparams,
        transform=False,
        mosaic_crop_size=320,
    )
    item = dataset.get_image_and_label(1)

    assert isinstance(
        item["instances"], uutils.instance.Instances
    ), "'instances' value should be of type Ultralytics Instances"

    expected_keys = ["im_file", "img", "cls", "ori_shape", "resized_shape"] 
    assert all(key in item for key in expected_keys), f"Missing key from: {expected_keys}"


def test_get_dataloader(remote_dir: str, local_dir, hyperparams: Hyperparameters):
    """Test the get_dataloader function."""
    dataloader = get_dataloader(local_dir, remote_dir, hyperparams, None)

    assert isinstance(dataloader, DataLoader)

    for batch in dataloader:
        assert (
            len(batch["im_file"]) == hyperparams.batch_size
        ), f"Batch size should be {hyperparams.batch_size}"

    shutil.rmtree(local_dir)


@pytest.fixture
def data() -> list[dict[str, Any]]:
    # Create dummy images (640x640 RGB image tensors)
    img1 = torch.rand(3, 640, 640)
    img2 = torch.rand(3, 640, 640)
    img3 = torch.rand(3, 640, 640)

    data = [
        {
            "im_file": "path/to/img1.jpg",
            "img": img1,
            "bboxes": np.array([0.4, 0.4, 0.6, 0.6, 0.2, 0.33, 0.1, 0.55]),
            "cls": np.array([0.0, 0.0]),
            "ori_shape": np.array([640, 640], dtype=np.uint32),
            "resized_shape": np.array([640, 640], dtype=np.uint32),
        },
        {
            "im_file": "path/to/img2.jpg",
            "img": img2,
            "bboxes": np.array(
                [0.12, 0.55, 0.78, 0.97, 0.03, 0.8, 0.1, 0.77, 0.49, 0.21, 0.66, 0.99]
            ),
            "cls": np.array([0.0, 0.0, 0.0]),
            "ori_shape": np.array([640, 640], dtype=np.uint32),
            "resized_shape": np.array([640, 640], dtype=np.uint32),
        },
        {
            "im_file": "path/to/img3.jpg",
            "img": img3,
            "bboxes": np.array([]),
            "cls": np.array([]),
            "ori_shape": np.array([640, 640], dtype=np.uint32),
            "resized_shape": np.array([640, 640], dtype=np.uint32),
        },
    ]
    return data


def test_collate_fn_img_bboxes(data):
    """Test bbox output of the collate_fn_img function."""
    batch = collate_fn_img(data)
    print(batch["bboxes"])
    assert torch.allclose(
        batch["bboxes"],
        torch.tensor(
            [
                [0.4, 0.4, 0.6, 0.6],
                [0.2, 0.33, 0.1, 0.55],
                [0.12, 0.55, 0.78, 0.97],
                [0.03, 0.8, 0.1, 0.77],
                [0.49, 0.21, 0.66, 0.99],
            ],
            dtype=torch.float64,
        ),
    )


def test_collate_fn_img_cls(data):
    """Test cls output of the collate_fn_img function."""
    batch = collate_fn_img(data)

    assert torch.allclose(
        batch["cls"],
        torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype=torch.float64),
    )


def test_collate_fn_img_img(data):
    """Test cls output of the collate_fn_img function."""
    batch = collate_fn_img(data)
    img = batch["img"]
    assert (3, 3, 640, 640) == tuple(
        img.shape
    ), "Shape should be (batch_size (3), channels (3), height (640), width (640))"


def test_collate_fn_img_ori_shape(data):
    """Test ori_shape output of the collate_fn_img function."""
    batch = collate_fn_img(data)
    ori_shapes = batch["ori_shape"]
    for ori_shape in ori_shapes:
        assert (640, 640) == tuple(ori_shape), "Original shapes should be 640x640"


def test_collate_fn_img_img_file(data):
    """Test im_file output of the collate_fn_img function."""
    batch = collate_fn_img(data)
    im_files = batch["im_file"]
    assert ("path/to/img1.jpg", "path/to/img2.jpg", "path/to/img3.jpg") == im_files