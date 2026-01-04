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


def test_rgb2hsv() -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255  # red
    hsv = imgviz.rgb2hsv(rgb)
    assert hsv.dtype == np.uint8
    assert hsv.shape == rgb.shape


def test_hsv2rgb() -> None:
    hsv = np.zeros((10, 10, 3), dtype=np.uint8)
    hsv[:, :, 2] = 255  # value
    rgb = imgviz.hsv2rgb(hsv)
    assert rgb.dtype == np.uint8
    assert rgb.shape == hsv.shape


def test_get_fg_color() -> None:
    from imgviz._color import get_fg_color

    # Dark background -> white foreground
    assert get_fg_color((0, 0, 0)) == (255, 255, 255)
    # Light background -> black foreground
    assert get_fg_color((255, 255, 255)) == (0, 0, 0)
