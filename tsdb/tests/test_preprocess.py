"""
This module tests code in tsdb.preprocessing.preprocess functions on image data. If needed,
function docstrings can include examples of what is being tested.
"""

import pytest
from tsdb.preprocessing.preprocess import data_augmentation


def test_data_augmentation():
    transforms = data_augmentation()
    assert isinstance(transforms, list), f"Expected a PySpark Column object, got {type(result)}"