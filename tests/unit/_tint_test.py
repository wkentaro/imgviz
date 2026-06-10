import numpy as np
import pytest
from cmap import ColorLike
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def rgb() -> NDArray[np.uint8]:
    return imgviz.data.arc2017()["rgb"]


def test_tint(rgb: NDArray[np.uint8]) -> None:
    res = imgviz.tint(rgb, "red", alpha=0.3)
    assert res.shape == rgb.shape
    assert res.dtype == rgb.dtype
    assert not np.array_equal(res, rgb)


def test_tint_alpha_zero_returns_input_unchanged(rgb: NDArray[np.uint8]) -> None:
    res = imgviz.tint(rgb, "red", alpha=0.0)
    assert np.array_equal(res, rgb)
    assert res is not rgb  # returns a copy, not the input


def test_tint_alpha_one_is_solid_color(rgb: NDArray[np.uint8]) -> None:
    res = imgviz.tint(rgb, (10, 20, 30), alpha=1.0)
    assert np.array_equal(res, np.full_like(rgb, [10, 20, 30]))


@pytest.mark.parametrize("color", ["red", "#00ff00", (0, 0, 255)])
def test_tint_accepts_color_formats(rgb: NDArray[np.uint8], color: ColorLike) -> None:
    res = imgviz.tint(rgb, color, alpha=0.5)
    assert res.shape == rgb.shape


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tint_float(dtype: type[np.floating]) -> None:
    img = np.random.rand(20, 20, 3).astype(dtype)
    res = imgviz.tint(img, "red", alpha=0.5)
    assert res.dtype == dtype
    assert res.min() >= 0 and res.max() <= 1
    assert np.array_equal(imgviz.tint(img, "red", alpha=0.0), img)


def test_tint_rejects_non_rgb() -> None:
    with pytest.raises(ValueError, match="must be RGB"):
        imgviz.tint(np.zeros((10, 10), dtype=np.uint8), "red")


def test_tint_rejects_invalid_dtype() -> None:
    with pytest.raises(ValueError, match="uint8 or float"):
        imgviz.tint(np.zeros((10, 10, 3), dtype=np.int32), "red")


@pytest.mark.parametrize("alpha", [-0.1, 1.5, float("nan")])
def test_tint_rejects_invalid_alpha(rgb: NDArray[np.uint8], alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be"):
        imgviz.tint(rgb, "red", alpha=alpha)
