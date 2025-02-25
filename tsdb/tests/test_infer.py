from typing import Any, Callable
from unittest.mock import Mock
from dataclasses import dataclass

import pytest
import torch
from PIL import Image
from pyspark.testing import assertSchemaEqual

from tsdb.ml.infer import make_towerscout_predict_udf, UDF_RETURN_TYPE
import tsdb.preprocessing.transformations as trf


@pytest.fixture
def batch_size() -> int:
    return 5


@pytest.fixture
def num_workers() -> int:
    return 2


@pytest.fixture
def catalog() -> str:
    return "edav_dev_csels"


@pytest.fixture
def schema() -> str:
    return "towerscout"


@pytest.fixture
def yolo_alias() -> str:
    return "testing"


@pytest.fixture
def efficientnet_alias() -> str:
    return "testing"


@pytest.fixture
def image_binary_dir() -> str:
    return "/Volumes/edav_dev_csels/towerscout/misc/unit_tests/image_binary_dataset/"


def test_predict(
    catalog: str,
    schema: str,
    yolo_alias: str,
    efficientnet_alias: str,
    batch_size: int,
    num_workers: int,
    image_binary_dir: str,
) -> None:
    """
    Tests the make_towerscout_predict_udf function.
    """
    towerscout_inference_udf = make_towerscout_predict_udf(
        catalog, schema, yolo_alias, efficientnet_alias, batch_size, num_workers
    )

    silver_df = (
        spark.read.format("delta")
        .table(f"{catalog}.{schema}.test_image_silver")
        .limit(2)
    )

    image_df = (
        spark.read.format("binaryFile")
        .load(image_binary_dir)
        .select("content")
        .limit(2)
    )

    transformed_df = (
        image_df.transform(trf.parse_file_path)
        .transform(trf.perform_inference, towerscout_inference_udf)
        .transform(trf.current_time)
        .transform(trf.hash_image)
        .selectExpr(
            "user_id",
            "request_id",
            "uuid",
            "processing_time",
            "results.bboxes as bboxes",
            "image_hash",
            "path as image_path",
            "results.model_version as model_version",
            "results.image_metadata as image_metadata",
            "results.map_provider as map_provider",
            "results.image_id as image_id",
        )
    )

    assert assertSchemaEqual(
        silver_df.schema, transformed_df.schema
    ), "Schema of bronze data transformed by predict UDF does not match silver table schema."