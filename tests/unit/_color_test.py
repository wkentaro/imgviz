from collections.abc import Callable

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


def test_asrgb_copy_does_not_alias_rgba_input(
    images: dict[str, NDArray[np.uint8]],
) -> None:
    rgba = images["rgba"]
    original = rgba[:, :, :3].copy()

    rgb = imgviz.asrgb(rgba, copy=True)
    rgb[:] = 0

    np.testing.assert_array_equal(rgba[:, :, :3], original)


def test_asrgb_no_copy_returns_view_for_rgba_input(
    images: dict[str, NDArray[np.uint8]],
) -> None:
    rgba = images["rgba"]

    rgb = imgviz.asrgb(rgba)

    assert np.shares_memory(rgb, rgba)


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


@pytest.mark.parametrize("convert", [imgviz.rgb2gray, imgviz.rgb2rgba])
@pytest.mark.parametrize(
    ("image", "match"),
    [
        (np.zeros((4, 4), dtype=np.uint8), "rgb must be 3 dimensional"),
        (np.zeros((4, 4, 4), dtype=np.uint8), r"rgb shape must be \(H, W, 3\)"),
        (np.zeros((4, 4, 3), dtype=np.float32), r"rgb dtype must be np\.uint8"),
    ],
    ids=["non-3d", "wrong-channels", "non-uint8"],
)
def test_rgb_converter_rejects_invalid(
    convert: Callable[[NDArray], NDArray[np.uint8]], image: NDArray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        convert(image)


@pytest.mark.parametrize(
    ("image", "match"),
    [
        (np.zeros((4, 4, 3), dtype=np.uint8), "gray must be 2 dimensional"),
        (np.zeros((4, 4), dtype=np.float32), r"gray dtype must be np\.uint8"),
    ],
    ids=["non-2d", "non-uint8"],
)
def test_gray2rgb_rejects_invalid(image: NDArray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        imgviz.gray2rgb(image)


@pytest.mark.parametrize(
    ("convert", "match"),
    [
        (imgviz.asgray, "unsupported image format to convert to gray"),
        (imgviz.asrgb, r"unsupported image format to convert to rgb\b"),
        (imgviz.asrgba, "unsupported image format to convert to rgba"),
    ],
    ids=["asgray", "asrgb", "asrgba"],
)
def test_as_converter_rejects_unsupported_shape(
    convert: Callable[[NDArray], NDArray[np.uint8]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        convert(np.zeros((4, 4, 2), dtype=np.uint8))
