from typing import Any
from unittest.mock import Mock
from dataclasses import dataclass

import pytest
import torch
from PIL import Image

from tsdb.ml.detections import YOLOv5_Detector


@dataclass
class RawDetection:
    xyxyn: torch.Tensor
    names: tuple[str] = ("ct",)


@pytest.fixture()
def sample_images():
    """
    A mock batch of data for the following tests.
    batch_idx corresponds to index of the image the bounding box is for in the im_file section of this dict.
    For example if batch_idx is 0 then the box is from the first image 'path/img1.jpg' in the im_file section.
    """
    images = [Image.new("RGB", (512, 512), color=(255, 255, 255)) for i in range(2)]

    return images


@pytest.fixture()
def sample_raw_detections():
    """
    A mock batch of data for the following tests.
    """
    raw_detections = [
    RawDetection(
        torch.tensor([[[0.3, 0.3, 0.5, 0.5, 0.5, 0], [0.1, 0.1, 0.2, 0.2, 0.8, 0]]])
    ),  # mock model output for first image
    RawDetection(
        torch.tensor([[[0.2, 0.2, 0.4, 0.4, 0.2, 0]]])
    ),  # mock model output for second image
    ]

    return raw_detections

def test_predict(sample_images: list[Image], sample_raw_detections: list[RawDetection]):
    # Mock EfficientNet model output
    mock_model: Mock = Mock()

    # Mock YOLOv5 model outputs (detections): [x1, y1, x2, y2, conf, label]
    mock_model.side_effect = sample_raw_detections

    # Instantiate EN_Classifier with the mock model
    detector = YOLOv5_Detector(model=mock_model, batch_size=1, uc_version="v1")

    detections = detector.predict(sample_images, secondary=None)

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
            assert type(box["secondary"]) == int, "Class labels from secondary model should be ints"