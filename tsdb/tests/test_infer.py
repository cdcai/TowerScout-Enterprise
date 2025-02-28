from typing import Any, Callable
from unittest.mock import Mock
from dataclasses import dataclass

import pytest
from pyspark.testing import assertSchemaEqual
from pyspark.sql import SparkSession
import torch
from PIL import Image

import tsdb.preprocessing.transformations as trf
from tsdb.ml.infer import (
    make_towerscout_predict_udf,
    UDF_RETURN_TYPE,
    inference_collate_fn,
    apply_secondary_model,
    parse_yolo_detections,
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
def batch_size() -> int:
    return 5


@pytest.fixture
def num_workers() -> int:
    return 2


@pytest.fixture
def yolo_alias() -> str:
    return "testing"


@pytest.fixture
def efficientnet_alias() -> str:
    return "testing"


@pytest.fixture
def sample_uncollated_batch() -> dict[str, Any]:
    uncollated_batch = [
        {
            "height": 640,
            "width": 640,
            "lat": 2.33,
            "long": -1.22,
            "image_id": 0,
            "map_provider": "Azure",
            "image": Image.new("RGB", (640, 640), (0, 0, 0)),
        },
        {
            "height": 640,
            "width": 640,
            "lat": -31.2,
            "long": 53.1,
            "image_id": 1,
            "map_provider": "Bing",
            "image": Image.new("RGB", (640, 640), (0, 0, 0)),
        },
    ]

    return uncollated_batch


@dataclass
class RawDetections:
    xyxyn: torch.Tensor
    names: tuple[str] = ("ct",)


@pytest.fixture()
def sample_images():
    """
    A mock batch of images for the following tests.
    """
    images = [Image.new("RGB", (512, 512), color=(255, 255, 255)) for i in range(2)]

    return images


@pytest.fixture()
def sample_raw_detections():
    """
    A mock batch of detections for the following tests.
    """
    raw_detections = RawDetections(
        [
            torch.tensor(
                [[0.3, 0.3, 0.5, 0.5, 0.5, 0], [0.1, 0.1, 0.2, 0.2, 0.8, 0]]
            ),  # mock model output (detections) for 1st image
            torch.tensor(
                [[0.2, 0.2, 0.4, 0.4, 0.2, 0]]
            ),  # mock model output (detections) for 2nd image
        ]
    )

    return raw_detections


def test_inference_collate_fn(sample_uncollated_batch: dict[str, Any]) -> None:
    """
    Test the inference_collate_fn function.
    """
    collated_batch = inference_collate_fn(sample_uncollated_batch)
    assert len(collated_batch["images"]) == len(
        sample_uncollated_batch
    ), "Expected the same number of images in the collated batch as the uncollated batch"
    assert (
        "image" not in collated_batch["images_metadata"]
    ), "Expected the 'image' key to be removed from the 'images_metadata' key of collated batch"


def test_apply_secondary_model():
    """
    Test the apply_secondary_model function. Test adapted from the test for
    the 'classify' method of the EN_Classifier object.
    """
    # Mock 'EfficientNet' model output
    mock_model: Mock = Mock()
    mock_model.side_effect = [
        torch.tensor([[0.1]]),  # Simulates output for the first detection
        torch.tensor([[0.8]]),  # Simulates output for the second detection
        torch.tensor([[0.95]]),  # Simulates output for the third detection
    ]

    # Create a dummy image (e.g., 512x512 white image)
    img: Image.Image = Image.new("RGB", (512, 512), color=(255, 255, 255))

    # Mock YOLOv5 detections: [x1, y1, x2, y2, conf, cls, label, custom_field]
    detections = [
        [0.2, 0.2, 0.4, 0.4, 0.2, 0, "non-tower", 0],  # Below min_conf
        [0.3, 0.3, 0.5, 0.5, 0.5, 0, "tower", 0],  # Within confidence range
        [0.1, 0.1, 0.2, 0.2, 0.8, 0, "tower", 0],  # Above max_conf
    ]

    apply_secondary_model(mock_model, img, detections, min_conf=0.25, max_conf=0.65)

    # Validate updated detections
    assert detections[0][-1] == 0  # Below min_conf
    assert 0 < detections[1][-1] < 1  # Within confidence range
    assert detections[2][-1] == 1  # Above max_conf


def test_parse_yolo_detections(
    sample_images: list[Image], sample_raw_detections: RawDetections
):
    """
    Test the parse_yolo_detections function. Test adapted from the test for
    the 'predict' method of the YOLOv5_Detector object.
    """
    # Mock YOLOv5 model output
    mock_model = Mock()

    # Mock YOLOv5 model outputs (detections): [x1, y1, x2, y2, conf, label]
    mock_model.side_effect = sample_raw_detections

    detections = parse_yolo_detections(
        images=sample_images, yolo_results=sample_raw_detections
    )

    assert len(detections) == 2, "Should contain detections for 2 images"

    for detection in detections:
        assert type(detection) == list, "Each detection should be a list"
        for box in detection:
            assert type(box["x1"]) == float, "Box coordinates should be floats"
            assert type(box["x2"]) == float, "Box coordinates should be floats"
            assert type(box["y1"]) == float, "Box coordinates should be floats"
            assert type(box["y2"]) == float, "Box coordinates should be floats"
            assert type(box["conf"]) == float, "Confidence should be floats"
            assert type(box["class"]) == int, "Class label should be ints"
            assert box["class_name"] == "ct", "Class names should be strings"
            assert (
                type(box["secondary"]) == int
            ), "Class labels from secondary model should be ints"


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
    towerscout_inference_udf = make_towerscout_predict_udf(
        catalog, schema, yolo_alias, efficientnet_alias, batch_size, num_workers
    )
    assert isinstance(
        towerscout_inference_udf, Callable
    ), "make_towerscout_predict_udf must return a Callable (Pandas UDF)"
    assert hasattr(towerscout_inference_udf, "returnType"), "UDF must have return type"
    assert (
        towerscout_inference_udf.returnType == UDF_RETURN_TYPE
    ), f"UDF return type must match that specified in {UDF_RETURN_TYPE}"


def test_predict(
    spark: SparkSession,
    catalog: str,
    schema: str,
    yolo_alias: str,
    efficientnet_alias: str,
    batch_size: int,
    num_workers: int,
    image_binary_dir: str,
) -> None:
    """
    Tests the predict UDF output by the make_towerscout_predict_udf function.
    We transform a dataframe containing bronze images using the predict UDF and then
    check to see if the resulting dataframes schema matches using PySparks `assertSchemaEqual`
    function. Note that we set ignoreNullable=True since some of the resulting columns
    in the transformed_df, such as `image_hash` have nullability set to False when the same column has
    nullability set to True in the silver_df. This discrepancy occurs even when using the
    old version of the towerscout_inference_udf function so it can be ignored.
    """
    towerscout_inference_udf = make_towerscout_predict_udf(
        catalog, schema, yolo_alias, efficientnet_alias, batch_size, num_workers
    )

    silver_df = (
        spark.read.format("delta")
        .table(f"{catalog}.{schema}.test_image_silver")
        .limit(2)
    )

    image_df = spark.read.format("binaryFile").load(image_binary_dir).limit(2)

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
            "results.image_id as image_id",
            "results.model_version as model_version",
            "results.image_metadata as image_metadata",
            "results.map_provider as map_provider",
        )
    )

    assertSchemaEqual(silver_df.schema, transformed_df.schema, ignoreNullable=True)