import pytest
import torch
from PIL import Image
from tsdb.ml.utils import YOLOv5Detection
from unittest.mock import Mock
from tsdb.ml.efficientnet import EN_Classifier 
from typing import List

def test_classify() -> None:
    """
    Test the classify method of the EN_Classifier.
    """
    # Mock EfficientNet model output
    mock_model: Mock = Mock()
    mock_model.side_effect = [
        torch.tensor([[0.1]]),  # Simulates output for the first detection
        torch.tensor([[0.8]]),  # Simulates output for the second detection
        torch.tensor([[0.95]])  # Simulates output for the third detection
    ]

    # Instantiate EN_Classifier with the mock model
    classifier: EN_Classifier = EN_Classifier(mock_model)

    # Create a dummy image (e.g., 512x512 white image)
    img: Image.Image = Image.new("RGB", (512, 512), color=(255, 255, 255))

    # Mock YOLOv5 detections: [x1, y1, x2, y2, conf, cls, label, custom_field]
    detections: List[List] = [
        [0.2, 0.2, 0.4, 0.4, 0.2, 0, "non-tower", 0],  # Below min_conf
        [0.3, 0.3, 0.5, 0.5, 0.5, 0, "tower", 0],      # Within confidence range
        [0.1, 0.1, 0.2, 0.2, 0.8, 0, "tower", 0],      # Above max_conf
    ]

    # Call classify
    classifier.classify(img, detections, min_conf=0.25, max_conf=0.65)

    # Validate updated detections
    assert detections[0][-1] == 0  # Below min_conf
    assert 0 < detections[1][-1] < 1  # Within confidence range
    assert detections[2][-1] == 1  # Above max_conf
