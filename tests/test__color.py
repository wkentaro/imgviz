import numpy as np

import imgviz


def test_asgray():
    data = imgviz.data.arc2017()
    gray = imgviz.asgray(data["rgb"])

    assert gray.ndim == 2
    assert gray.dtype == np.uint8


def test_asrgb():
    data = imgviz.data.arc2017()
    gray = imgviz.rgb2gray(data["rgb"])

    rgb = imgviz.asrgb(gray)

    assert rgb.dtype == np.uint8
    assert rgb.ndim == 3


def test_asrgba():
    data = imgviz.data.arc2017()
    gray = imgviz.rgb2gray(data["rgb"])

    rgba = imgviz.asrgba(gray)

    assert rgba.dtype == np.uint8
    assert rgba.ndim == 3
    assert rgba.shape[2] == 4
