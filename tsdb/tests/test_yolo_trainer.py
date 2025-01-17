from typing import Any

import pytest
import torch

from ultralytics.utils import IterableSimpleNamespace
from torch import tensor

@pytest.fixture()
def lb() -> list[torch.Tensor]:
    return []


@pytest.fixture()
def args() -> list[str]:
    """
    A mock args object for the following tests.
    We specify the threshold confidence score of 0.5 here 
    for testing the filtering feature of the postprocess function
    """
    return IterableSimpleNamespace(
        conf=0.5, iou=0.3, single_cls=True, max_det=300, save_hybrid=False
    )


@pytest.fixture()
def sample_batch():
    """
    A mock batch of data for the following tests.
    batch_idx corresponds to index of the image the bounding box is for in the im_file section of this dict.
    For example if batch_idx is 0 then the box is from the first image 'path/img1.jpg' in the im_file section.
    """
    shape = (2, 3, 1500, 1500)  # To create a random tensor (image) with the given shape
    batch = {
        "im_file": (
            "path/img1.jpg",
            "path/img2.jpg",
        ),
        "ori_shape": ((1500, 1500), (1500, 1500)),
        "resized_shape": ((1500, 1500), (1500, 1500)),
        "ratio_pad": (None, None),
        "img": torch.randint(0, 256, shape, dtype=torch.uint8),
        "cls": tensor([[0.0], [0.0], [0.0], [0.0], [0.0]]),  # class labels
        "bboxes": tensor(
            [
                [0.5746, 0.6375, 0.2610, 0.3689],
                [0.3660, 0.6481, 0.1675, 0.3164],
                [0.5915, 0.5939, 0.1315, 0.1461],
                [0.4127, 0.5856, 0.1139, 0.1259],
                [0.3695, 0.7020, 0.0239, 0.0671],
            ]
        ),
        "batch_idx": tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    }

    return batch


@pytest.fixture()
def pred_prepared():
    """
    A mock prediction tensor for the following tests. This is the "prepared" version that is
    returned by the _prepare_pred function when you feed it the pred_unprepared tensor.
    The confidence score is the 5th element in the box format and will be used in the postprocess
    function test to test the filtering functionality of the postprocess function.
    Note that these are also the true bounding boxes that are in the batch defined above
    *after* they have been scaled and padded by the _prepare_batch function. 
    """
    # box format: (x1, y1, x2, y2, confidence, class)
    pred = tensor(
        [
            [211.1375, 366.8750, 336.7625, 604.1750, 0.9, 0.0],  # Box 1 (should be kept)
            [393.7625, 390.0875, 492.3875, 499.6625, 0.77, 0.0],  # Box 2 (should be kept)
            [266.2625, 391.4375, 351.6875, 485.8625, 0.6, 0.0],  # Box 3 (should be kept)
            [267.6125, 500.7875, 285.5375, 551.112, 0.39, 0.0], # Box 4 (should be filtered out)
        ]  
    ).unsqueeze(0)  # Adding batch dimension

    return pred


@pytest.fixture()
def pred_unprepared(pred_prepared: torch.Tensor, sample_batch: torch.Tensor, si: int):
    """
    A mock prediction tensor for the following tests. This is the "unprepared" version. To create
    the unprepared version I simply perform the inverse of the operations that the _prepare_pred
    function performs on the input tensor it is given. These operations are determined by the
    ratio_pad element of the sample_batch dict. The ratio_pad element is a tuple of two elements.
    The first tuple gives you the gain which is used in the _prepare_batch function
    (the scale_boxes function that it calls divides all boxes by this value) and the second element gives you
    you the two numbers to pad the boxes by (the scale_boxes function that it calls subtracts pad[0] and
    pad[1] from the respective corners of the bounding boxes)
    """

    pred = pred_prepared.clone()
    #ratio_pad = sample_batch["ratio_pad"][si]
    img1_shape = sample_batch["ori_shape"][si]
    img0_shape = sample_batch["resized_shape"][si]
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new 
    pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )
    pred[:, :, :4] += pad[0] / 2
    pred[:, :, :4] *= gain

    return pred


@pytest.fixture()
def si() -> int:
    """
    This is used to selected the bounding boxes and their associated labels 
    that correspond to image 0 in the sample_batch dict.
    """
    return 0


@pytest.fixture()
def device() -> str:
    return "cpu"