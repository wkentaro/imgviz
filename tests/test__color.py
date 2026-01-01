import numpy as np

import imgviz


def test_asgray():
    data = imgviz.data.arc2017()

    # gray input
    gray = imgviz.rgb2gray(data["rgb"])
    result = imgviz.asgray(gray)
    assert result.ndim == 2
    assert result.dtype == np.uint8

    # rgb input
    rgb = data["rgb"]
    result = imgviz.asgray(rgb)
    assert result.ndim == 2
    assert result.dtype == np.uint8

    # rgba input
    rgba = imgviz.rgb2rgba(data["rgb"])
    result = imgviz.asgray(rgba)
    assert result.ndim == 2
    assert result.dtype == np.uint8


def test_asrgb():
    data = imgviz.data.arc2017()

    # gray input
    gray = imgviz.rgb2gray(data["rgb"])
    result = imgviz.asrgb(gray)
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 3

    # rgb input
    rgb = data["rgb"]
    result = imgviz.asrgb(rgb)
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 3

    # rgba input
    rgba = imgviz.rgb2rgba(data["rgb"])
    result = imgviz.asrgb(rgba)
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_asrgba():
    data = imgviz.data.arc2017()

    # gray input
    gray = imgviz.rgb2gray(data["rgb"])
    result = imgviz.asrgba(gray)
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 4

    # rgb input
    rgb = data["rgb"]
    result = imgviz.asrgba(rgb)
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 4

    # rgba input
    rgba = imgviz.rgb2rgba(data["rgb"])
    result = imgviz.asrgba(rgba)
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 4
