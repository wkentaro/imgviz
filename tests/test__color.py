import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def images() -> dict[str, NDArray[np.uint8] | NDArray[np.bool_]]:
    data = imgviz.data.arc2017()
    rgb = data["rgb"]
    return {
        "bool": imgviz.rgb2gray(rgb) > 128,
        "gray": imgviz.rgb2gray(rgb),
        "rgb": rgb,
        "rgba": imgviz.rgb2rgba(rgb),
    }


def test_asgray(images: dict[str, NDArray[np.uint8]]) -> None:
    for img in images.values():
        result = imgviz.asgray(img)
        assert result.ndim == 2
        assert result.dtype == np.uint8


def test_asrgb(images: dict[str, NDArray[np.uint8]]) -> None:
    for img in images.values():
        result = imgviz.asrgb(img)
        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3


def test_asrgba(images: dict[str, NDArray[np.uint8]]) -> None:
    for img in images.values():
        result = imgviz.asrgba(img)
        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 4
