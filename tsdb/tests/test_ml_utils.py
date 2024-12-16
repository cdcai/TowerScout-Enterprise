import pytest
import numpy as np

from PIL import Image
from tsdb.ml.utils import cut_square_detection, get_model_tags


@pytest.fixture
def mock_img() -> Image.Image:
    # create a blank image array
    img_arr = np.zeros((640, 640), dtype=np.uint8)

    # add a black filled rectangle to the image array
    img_arr[80:320, 120:200] = 255

    # mode 'L' is used for grayscale, nxn array each containing val 0-255
    return Image.fromarray(img_arr, mode='L')


@pytest.fixture
def mock_img_corner() -> Image.Image:
    # create a blank image array
    img_arr = np.zeros((640, 640), dtype=np.uint8)

    # add a black filled rectangle to the image array, place it in a corner
    img_arr[0:120, 0:240] = 255

    # mode 'L' is used for grayscale, nxn array each containing val 0-255
    return Image.fromarray(img_arr, mode='L')


def test_perfect_squareness(mock_img: Image.Image) -> None:
    '''
    If the cropped region is close enough to the center of the image,
    the resulting Image should be perfectly square
    '''
    x1 = 0.4
    y1 = 0.4
    x2 = 0.6
    y2 = 0.6

    res = cut_square_detection(mock_img, x1=x1, y1=y1, x2=x2, y2=y2)
    w_res, h_res = res.size

    assert w_res == h_res, \
        "Resulting width and height should match for perfect square test"


def test_invalid_inputs(mock_img: Image.Image) -> None:
    '''
    cut_square_detection() should raise an error if the input coordinates
    are invalid
    '''
    with pytest.raises(ValueError) as e:
        res = cut_square_detection(mock_img, x1=0.9, y1=0.8, x2=0.3, y2=0.5)
        assert str(e.value) == "Coordinate 'right' is less than 'left'"


def test_full_region_capture(mock_img: Image.Image) -> None:
    '''
    The mock_img contains a black filled rectangle. If the input
    coordinates outline this rectangle, the resulting Image should
    contain the full original rectangle, plus a bordering buffer.
    '''
    # calculate proper floats based on mock_img values,
    # these should perfectly outline the black rectangle
    x1 = 80/640
    x2 = 320/640
    y1 = 120/640
    y2 = 200/640

    res = cut_square_detection(mock_img, x1=x1, y1=y1, x2=x2, y2=y2)

    # calculate expected number of black pixels
    orig_black = int(np.sum(mock_img.getdata())/255)
    res_black = int(np.sum(res.getdata())/255)

    assert orig_black == res_black, \
        "Output Image should capture all black pixels from input"

    # and confirm that at least some buffer was added after the crop
    w, h = res.size
    total_pixels = w * h
    assert total_pixels > orig_black, \
        "Output Image should have more pixels than input due to addition of buffer"


def test_target_on_border(mock_img_corner: Image.Image) -> None:
    '''
    If the target crop section is on touching at least one edge of
    the image, the output Image should contain the full original
    contents, plus a bordering buffer.

    mock_img_corner contains a black filled rectangle originating
    at coordinates (0, 0)
    '''
    x1 = 0
    x2 = 120/640
    y1 = 0
    y2 = 240/640

    res = cut_square_detection(mock_img_corner, x1=x1, y1=y1, x2=x2, y2=y2)

    # calculate expected number of black pixels
    orig_black = int(np.sum(mock_img_corner.getdata())/255)
    res_black = int(np.sum(res.getdata())/255)

    assert orig_black == res_black, \
        "Output Image should capture all black pixels from input"

    # and confirm that at least some buffer was added after the crop
    w, h = res.size
    total_pixels = w * h
    assert total_pixels > orig_black, \
        "Output Image should have more pixels than input due to addition of buffer"
