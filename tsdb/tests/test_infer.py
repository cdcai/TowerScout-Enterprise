from typing import Any, Callable
from unittest.mock import Mock
from dataclasses import dataclass

import pytest
import torch
from PIL import Image

from tsdb.ml.infer import make_towerscout_predict_udf, UDF_RETURN_TYPE


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


def test_make_towerscout_predict_udf(
    catalog: str,
    schema: str,
    yolo_alias: str,
    efficientnet_alias: str,
    batch_size: int,
    num_workers: int,
) -> None:
    """
    Tests the make_towerscout_predict_udf function.
    """
    towerscout_inference_udf = make_towerscout_predict_udf(catalog, schema, yolo_alias, efficientnet_alias, batch_size, num_workers)
    assert isinstance(towerscout_inference_udf, Callable), "make_towerscout_predict_udf must return a Callable (Pandas UDF)"
    assert hasattr(towerscout_inference_udf, "returnType"), "UDF must have return type"
    assert towerscout_inference_udf.returnType == UDF_RETURN_TYPE, f"UDF return type must match that specified in {UDF_RETURN_TYPE}"