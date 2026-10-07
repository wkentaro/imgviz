import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz
from imgviz._colorblind import Kind

KINDS: list[Kind] = ["protanopia", "deuteranopia", "tritanopia"]


@pytest.fixture
def rgb() -> NDArray[np.uint8]:
    return imgviz.data.arc2017()["rgb"]


@pytest.mark.parametrize("kind", KINDS)
def test_colorblind(rgb: NDArray[np.uint8], kind: Kind) -> None:
    res = imgviz.colorblind(rgb, kind=kind)
    assert res.shape == rgb.shape
    assert res.dtype == rgb.dtype
    assert not np.array_equal(res, rgb)


@pytest.mark.parametrize("kind", KINDS)
def test_colorblind_is_noop_on_gray(kind: Kind) -> None:
    gray = np.repeat(
        np.random.randint(0, 256, size=(20, 20, 1), dtype=np.uint8), 3, axis=2
    )
    res = imgviz.colorblind(gray, kind=kind)
    assert np.array_equal(res, gray)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_colorblind_float(dtype: type[np.floating], kind: Kind) -> None:
    img = np.random.rand(20, 20, 3).astype(dtype)
    res = imgviz.colorblind(img, kind=kind)
    assert res.dtype == dtype
    assert res.min() >= 0 and res.max() <= 1

    gray = np.repeat(np.random.rand(20, 20, 1).astype(dtype), 3, axis=2)
    assert np.allclose(imgviz.colorblind(gray, kind=kind), gray)


@pytest.mark.parametrize("kind", KINDS)
def test_colorblind_applies_kind_specific_matrix(kind: Kind) -> None:
    px = np.array([[[200, 130, 60]]], dtype=np.uint8)
    expected = {
        "protanopia": [170, 169, 77],
        "deuteranopia": [174, 179, 81],
        "tritanopia": [196, 90, 93],
    }
    res = imgviz.colorblind(px, kind=kind)
    np.testing.assert_array_equal(res[0, 0], expected[kind])


def test_colorblind_rejects_invalid_kind(rgb: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        imgviz.colorblind(rgb, kind="monochromacy")  # type: ignore[arg-type]


@pytest.mark.parametrize("shape", [(20, 20), (20, 20, 4)])
def test_colorblind_rejects_non_rgb(shape: tuple[int, ...]) -> None:
    img = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="must be RGB"):
        imgviz.colorblind(img)


def test_colorblind_rejects_invalid_dtype() -> None:
    img = np.zeros((20, 20, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="uint8 or float"):
        imgviz.colorblind(img)
