import shutil
from typing import Any
import uuid

import pytest
import torch
import numpy as np

import ultralytics.utils as uutils

from tsdb.ml.types import Hyperparameters
from tsdb.ml.datasets import (
    get_dataloader,
    collate_fn_img,
    DataLoader,
    DataLoaders,
    YoloDataset,
)


@pytest.fixture
def remote_dir() -> str:
    return "/Volumes/edav_dev_csels/towerscout/misc/mosaic_streaming_unit_test/"


@pytest.fixture
def local_dir() -> str:
    # append uuid to avoid uuusing same cache between tests
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


def test_get_image_and_label(remote_dir: str, local_dir, hyperparams: Hyperparameters):
    dataset = YoloDataset(
        remote=remote_dir,
        local=local_dir,
        shuffle=True,
        hyperparameters=hyperparams,
        transform=False,
        image_size=320,
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