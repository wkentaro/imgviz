import numpy as np
import pytest

import imgviz


@pytest.fixture
def images():
    data = imgviz.data.arc2017()
    rgb = data["rgb"]
    return {
        "gray": imgviz.rgb2gray(rgb),
        "rgb": rgb,
        "rgba": imgviz.rgb2rgba(rgb),
    }


def test_asgray(images):
    for img in images.values():
        result = imgviz.asgray(img)
        assert result.ndim == 2
        assert result.dtype == np.uint8


def test_asrgb(images):
    for img in images.values():
        result = imgviz.asrgb(img)
        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3


def test_asrgba(images):
    for img in images.values():
        result = imgviz.asrgba(img)
        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 4
