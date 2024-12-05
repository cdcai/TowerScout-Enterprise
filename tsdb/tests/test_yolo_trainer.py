import pytest

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