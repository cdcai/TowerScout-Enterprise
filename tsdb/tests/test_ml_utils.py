import pytest
import numpy as np

from PIL import Image
from tsdb.ml.utils import cut_square_detection, get_model_tags


@pytest.fixture
def mock_img() -> Image.Image:
    # create a blank image array
    img_arr = np.zeros((640, 640), dtype=int)

    # add a black filled rectangle to the image array
    img_arr[80:320, 120:200] = 255

    return Image.fromarray(img_arr, mode='L')


def test_cut_square_detection(img: Image.Image) -> None:
    # test for correct width and height
    # test pixel contents
    # test edge cases, like invalid coordinates

    # calculate correct final width and height based on values from mock
    w_final = 320-80
    h_final = 200-120

    # calculate float values based on rectangle created in mock
    x1 = 80/640
    y1 = 120/640
    x2 = 320/640
    y2 = 200/640

    res = cut_square_detection(img, x1=x1, y1=y1, x2=x2, y2=y2)
    w_res, h_res = res.size

    assert w_res == w_final, "Resulting width should match calculated width"
    assert h_res == h_final, "Resulting height should match calculated height"

    res_array = np.array(res)
    all_black = True
    for i in range(len(res_array)):
        for j in range(len(res_array[i])):
            if res_array[i][j] != 256:
                all_black = False
                break
        if not all_black:
            break
    
    assert all_black, "Resulting image should be all black"


# def test_get_model_tags() -> None:
#     pass


# if __name__ == '__main__':
#     # create a blank image array
#     img_arr = np.zeros((640, 640), dtype=int)

#     # add a black filled rectangle to the image array
#     img_arr[80:320, 120:200] = 255

#     img = Image.fromarray(img_arr, mode='L')
#     print(cut_square_detection(img, 0.1, 0.2, 0.5, 0.8).size)
