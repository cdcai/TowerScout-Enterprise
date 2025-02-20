from typing import Any

import pytest
from PIL import Image

from tsdb.ml.infer import inference_collate_fn, apply_secondary_model


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

def test_inference_collate_fn(uncollated_batch: dict[str, Any]) -> None:
    collated_batch = inference_collate_fn(uncollated_batch)
    assert len(collated_batch['images']) ==  len(collated_batch), "Expected the same number of images in the collated batch as the uncollated batch"
    assert "image" not in collated_batch["images_metadata"], "Expected the 'image' key to be removed from the 'images_metadata' key of collated batch"

