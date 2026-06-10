from typing import Literal

import cmap
import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz
from imgviz._scalebar import _pick_nice_length


@pytest.fixture
def rgb() -> NDArray[np.uint8]:
    return imgviz.data.arc2017()["rgb"]


def test_scalebar(rgb: NDArray[np.uint8]) -> None:
    res = imgviz.scalebar(rgb, pixels_per_unit=200, unit="m")
    assert res.shape == rgb.shape
    assert res.dtype == rgb.dtype
    assert not np.array_equal(res, rgb)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (1.0, 1.0),
        (1.9, 1.0),
        (2.0, 2.0),
        (4.9, 2.0),
        (5.0, 5.0),
        (9.9, 5.0),
        (37.0, 20.0),
        (250.0, 200.0),
        (0.03, 0.02),
        (7e-9, 5e-9),
        (8e6, 5e6),
    ],
)
def test_pick_nice_length(target: float, expected: float) -> None:
    result = _pick_nice_length(target)
    assert result == pytest.approx(expected)
    assert result <= target  # never exceeds the budget it is given


@pytest.mark.parametrize("pixels_per_unit", [0.5, 5, 50, 500, 5000, 1e6])
def test_scalebar_runs_across_decades(
    rgb: NDArray[np.uint8], pixels_per_unit: float
) -> None:
    res = imgviz.scalebar(rgb, pixels_per_unit=pixels_per_unit)
    assert res.shape == rgb.shape
    assert res.dtype == rgb.dtype
    assert not np.array_equal(res, rgb)


@pytest.mark.parametrize("loc", ["lt", "rt", "lb", "rb"])
def test_scalebar_all_corners(
    rgb: NDArray[np.uint8], loc: Literal["lt", "rt", "lb", "rb"]
) -> None:
    res = imgviz.scalebar(rgb, pixels_per_unit=200, loc=loc)
    h, w = rgb.shape[:2]
    quadrant = {
        "lt": np.s_[: h // 2, : w // 2],
        "rt": np.s_[: h // 2, w // 2 :],
        "lb": np.s_[h // 2 :, : w // 2],
        "rb": np.s_[h // 2 :, w // 2 :],
    }[loc]
    assert not np.array_equal(res[quadrant], rgb[quadrant])  # drawn in that corner


def test_scalebar_auto_contrast_picks_white_on_dark() -> None:
    dark = np.zeros((200, 400, 3), dtype=np.uint8)
    res = imgviz.scalebar(dark, pixels_per_unit=50, color="auto")
    assert res.max() == 255  # white bar/label on black


def test_scalebar_auto_contrast_picks_black_on_light() -> None:
    light = np.full((200, 400, 3), 255, dtype=np.uint8)
    res = imgviz.scalebar(light, pixels_per_unit=50, color="auto")
    assert res.min() == 0  # black bar/label on white


@pytest.mark.parametrize("color", ["red", (255, 0, 0), np.array([255, 0, 0])])
def test_scalebar_explicit_color(rgb: NDArray[np.uint8], color: cmap.ColorLike) -> None:
    res = imgviz.scalebar(rgb, pixels_per_unit=200, color=color)
    assert (res == [255, 0, 0]).all(axis=2).any()  # some pure-red pixels drawn


def test_scalebar_rejects_non_3d_image() -> None:
    with pytest.raises(ValueError, match="uint8 RGB"):
        imgviz.scalebar(np.zeros((10, 10), dtype=np.uint8), pixels_per_unit=1)


def test_scalebar_rejects_non_uint8_image() -> None:
    with pytest.raises(ValueError, match="uint8 RGB"):
        imgviz.scalebar(np.zeros((10, 10, 3), dtype=np.float32), pixels_per_unit=1)


def test_scalebar_rejects_nonpositive_ppu(rgb: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match="pixels_per_unit"):
        imgviz.scalebar(rgb, pixels_per_unit=0)


def test_scalebar_rejects_bad_loc(rgb: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match="loc must be"):
        imgviz.scalebar(rgb, pixels_per_unit=200, loc="middle")  # type: ignore[arg-type]
