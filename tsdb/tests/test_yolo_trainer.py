from typing import Any

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
    """
    A mock args object for the following tests.
    We specidify the threshold confidence score of 0.5 here 
    for testing the filtering feature of the postprocess function
    """
    return IterableSimpleNamespace(
        conf=0.5, iou=0.3, single_cls=True, max_det=300, save_hybrid=False
    )


@pytest.fixture()
def sample_batch():
    """
    A mock batch of data for the following tests.
    batch_idx corresponds to index of the image the box is from in the im_file section of this dict.
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
        "ratio_pad": (((2.0,), (1.1, 1.1)), ((2.0,), (1.1, 1.1))),
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
        "batch_idx": tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
    }

    return batch


@pytest.fixture()
def pred_prepared():
    """
    A mock prediction tensor for the following tests. This is the "prepared" version that is
    returned by the _prepare_pred function when you feed it the pred_unprepared tensor.
    The confidence score is the 5th element in the box format and will be used in the postprocess
    function test to test the filtering functionality of the postprocess function.
    """
    # box format: (x1, y1, x2, y2, confidence, class)
    pred = tensor(
        [
            [211.1375, 366.8750, 336.7625, 604.1750, 0.9, 0.0],  # Box 1 (should be kept)
            [393.7625, 390.0875, 492.3875, 499.6625, 0.77, 0.0],  # Box 2 (should be kept)
            [266.2625, 391.4375, 351.6875, 485.8625, 0.6, 0.0],  # Box 3 (should be kept)
            [267.6125, 500.7875, 285.5375, 551.112, 0.39, 0.05], # Box 4 (should be filtered out)
        ]  
    ).unsqueeze(0)  # Adding batch dimension

    return pred


@pytest.fixture()
def pred_unprepared(pred_prepared: Tensor, sample_batch: Tensor, si: int):
    """
    A mock prediction tensor for the following tests. This is the "unprepared" version. To create
    the unprepared version I simply perform the inverse of the operations that the _prepare_pred
    function performs on the pred_prepared tensor. These operations are determined by the
    ratio_pad element of the sample_batch dict. The ratio_pad element is a tuple of two elements.
    The first tuple gives you the gain which is used in the _prepare_batch function
    (the scale_boxes function that it calls divides all boxes by this value) and the second element gives you
    you the two numbers to pad the boxes by (the scale_boxes function that it calls subtracts pad[0] and
    pad[1] from the respetvice corners of the bounding boxes)
    """

    pred = pred_prepared.clone()
    ratio_pad = sample_batch["ratio_pad"][si]
    gain = ratio_pad[0][0]
    pad = ratio_pad[1][0]
    pred[:, :, :4] += pad / 2
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


def test_postprocess(
    pred_unprepared: Tensor, args: IterableSimpleNamespace, lb: list[Tensor]
) -> None:
    """
    Tests the postprocess function from Ultralytics whose primary
    purpose is to filter out bounding boxes whose confidence score
    do not meet a particular threshold. For this test the confidence
    threshold is set to 0.5 so 1/4 of the boxes should be filtered.
    """

    output = postprocess(pred_unprepared, args, lb)

    assert isinstance(output, list), "Output should be a list"
    assert (
        len(output) == pred_unprepared.shape[0]
    ), "Output list length should match batch size"

    for out in output:
        assert isinstance(out, Tensor), "Each output element should be a tensor"
        if len(out) > 0:
            assert (
                out[:, 4] >= args.conf
            ).all(), "All confidences should be above or equal to the threshold"

            assert len(out) == 3, "Output should contain 75% of the original boxes (3)"


def test_prepare_batch(si: int, sample_batch: Tensor, device: str) -> None:
    """
    Tests the _prepare_batch function from Ultralytics whose primary
    purpose is to prepare a batch of images and labels given batch from the YOLO dataloader based on the
    the `si` arguement which tells use which image's bounding boxes and labels we want to retrieve from the YOLO
    dataloader batch. 
    For this test the batch index is set to 0 so only 4 out of the 5
    bounding boxes and their associated labels in the batch should be returned.
    """

    pbatch = _prepare_batch(si, sample_batch, device)

    assert torch.equal(
        pbatch["cls"], tensor([0.0, 0.0, 0.0, 0.0])
    ), "Only 4/5 labels should be returned"
    assert (
        pbatch["ratio_pad"] == sample_batch["ratio_pad"][si]
    ), "Only 4/5 ratio_pad's should be returned"


def test_prepare_pred(
    pred_unprepared: Tensor,
    pred_prepared: Tensor,
    sample_batch: dict[str, Any],
    si: int,
    device: str,
) -> None:
    """
    Tests the _prepare_pred function from Ultralytics whose primary
    purpose is to prepare a set of predicted bounding boxes by
    scaling the bounding boxes according to the values in 
    the `ratio_pad` tuple corresponding to the image with
    the index `si` in the batch.
    """
    pbatch = _prepare_batch(si, sample_batch, device)
    predn = _prepare_pred(pred_unprepared, pbatch)
    assert torch.allclose(predn, pred_prepared, atol=1e-4)


def test_score(
    sample_batch: dict[str, Any],
    pred_unprepared: Tensor,
    args: IterableSimpleNamespace,
    device: str,
):
    """
    Tests the score function.

    This test case should return {'accuracy_VAL': 0.75, 'f1_VAL': 0.8571428571428571}
    because the confusion matrix should be:
                      true ct,    true background
    predicted ct:        [[3,         0]
    predicted background: [1,         0]]

    This is because the first 3 predicted bounding box classes are correct and while the 4th one is also right
    it's confidence does not meet the required threshold of 0.5 (unlike the other 3 boxes) so it gets filtered out in the postprocess()
    step leading to a false negative for the fourth bounding box [0.4127, 0.5856, 0.1139, 0.1259] (undetected ct).
    Note that there are 5 bounding boxes in the batch but we only use 4 of them in this test case because
    we are testing a prediction for just image 0 not image 1 and the 1st bounding box in the batch is for image 1.
    """

    scores = score(
        minibatch=sample_batch,
        preds=pred_unprepared,
        step="VAL",
        device=device,
        args=args,
    )

    f1 = scores["f1_VAL"]
    acc = scores["accuracy_VAL"]
    absolute_tolerance = 0.0001
    assert pytest.approx(f1, abs=absolute_tolerance) == 0.8571428571428571
    assert pytest.approx(acc, abs=absolute_tolerance) == 0.75