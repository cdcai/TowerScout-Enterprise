import pytest
import torch

from ultralytics.utils import IterableSimpleNamespace
from torch import Tensor, tensor

from tsdb.ml.yolo_trainer import _prepare_batch, _prepare_pred, postprocess, score


@pytest.fixture()
def lb() -> list[Tensor]:
    return []


@pytest.fixture()
def args() -> list[str]:
    args = IterableSimpleNamespace(conf=0.5, iou=0.0, single_cls=True, max_det=300)
    return args


@pytest.fixture()
def sample_prediction():
    # box format: (x1, y1, x2, y2, confidence, class)
    prediction = tensor(
        [
            [0.508, 0.141, 0.27, 0.4, 0.9, 0],  # Box 1 (keep)
            [0.31, -0.42, 0.443, 0.5, 0.6, 0],  # Box 2 (keep)
            [0.92, 0.442, -0.43, 0.5, 0.46, 0],  # Box 3 (filter)
            [0.444, 0.2, 0.3, 0.5, 0.39, 0],  # Box 4 (filter)
        ]
    ).unsqueeze(0)  # Adding batch dimension

    return prediction


@pytest.fixture()
def si() -> int:
    return 1

@pytest.fixture()
def device() -> str:
    return "cpu"

@pytest.fixture()
def sample_batch():

    shape = (2, 3, 640, 640) # To create a random tensor (image) with the given shape
    batch = {
        "im_file": (
            "path/img1.jpg",
            "path/img2.jpg",
        ),
        "ori_shape": ((500, 381), (478, 640)),
        "resized_shape": ((640, 640), (640, 640)),
        "ratio_pad": (((1.28, 1.2808398950131235), (76, 0)), ((1.0, 1.0), (0, 81))),
        "img": torch.randint(0, 256, shape, dtype=torch.uint8),
        "cls": tensor(
            [[17.0], [17.0], [0.0], [0.0], [58.0]]
        ),
        "bboxes": tensor(
            [
                [0.5746, 0.6375, 0.2610, 0.3689],
                [0.3660, 0.6481, 0.1675, 0.3164],
                [0.5915, 0.5939, 0.1315, 0.1461],
                [0.4127, 0.5856, 0.1139, 0.1259],
                [0.3695, 0.7020, 0.0239, 0.0671]
            ]
        ),
        "batch_idx": tensor([0.0, 1.0, 0.0, 1.0, 1.0]) # batch_idx corresponds to index of the image the box is from in the im_file section of this dict
    }

    return batch


def test_postprocess(
    sample_prediction: Tensor, args: IterableSimpleNamespace, lb: list[Tensor]
) -> None:
    """
    Tests the postprocess function from Ultralytics whose primary
    purpose is to filter out bounding boxes whose confidence score
    do not meet a particular threshold. For this test the confidence
    threshold is set to 0.5 so 2/4 of the boxes should be filtered.
    """

    output = postprocess(sample_prediction, args, lb)

    assert isinstance(output, list), "Output should be a list"
    assert (
        len(output) == sample_prediction.shape[0]
    ), "Output list length should match batch size"

    for out in output:
        assert isinstance(out, Tensor), "Each output element should be a tensor"
        if len(out) > 0:
            assert (
                out[:, 4] >= args.conf
            ).all(), "All confidences should be above or equal to the threshold"

            assert len(out) == 2, "Output should contain 50% of the original boxes (2)"


def test_prepare_batch(
    si:int, sample_batch: Tensor, device: str
) -> None:
    """
    Tests the _prepare_batch function from Ultralytics whose primary
    purpose is to prepare a batch of images and labels given batch from the YOLO dataloader based on the 
    the si arguement which tells use which images bounding boxes and labels we want to retrieve from the YOLO
    dataloader batch. 
    For this test the batch index is set to 1 so only 3 out of the 5
    bounding boxes in the batch should be returned.
    """

    pbatch = _prepare_batch(si, sample_batch, device)

    assert torch.equal(pbatch['cls'], tensor([17., 0., 58.]))
    assert pbatch['ratio_pad'] == sample_batch['ratio_pad'][si]
