from typing import Any
from unittest.mock import Mock
from dataclasses import dataclass

import pytest
import torch
from PIL import Image

from tsdb.ml.infer import (
    inference_collate_fn,
    apply_secondary_model,
    parse_yolo_detections,
)


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
    # Mock EfficientNet model output
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

    # Call classify
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