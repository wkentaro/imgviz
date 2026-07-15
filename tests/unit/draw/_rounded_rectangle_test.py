import numpy as np
import PIL.Image
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def white_image() -> NDArray[np.uint8]:
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def test_rounded_rectangle(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.rounded_rectangle(
        white_image, (10, 10), (90, 90), radius=20, outline=(0, 0, 0), width=2
    )
    assert res.shape == white_image.shape
    assert res.dtype == white_image.dtype
    assert not np.array_equal(res, white_image)


def test_rounded_rectangle_rejects_missing_fill_and_outline(
    white_image: NDArray[np.uint8],
) -> None:
    with pytest.raises(ValueError, match="at least one of `fill` or `outline`"):
        imgviz.draw.rounded_rectangle(white_image, (0, 0), (50, 50), radius=10)


def test_rounded_rectangle_clips_the_corners(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.rounded_rectangle(
        white_image, (10, 10), (90, 90), radius=20, fill=(0, 0, 0)
    )

    # the bbox corner is cut away by the arc; the body is filled
    np.testing.assert_array_equal(res[10, 10], [255, 255, 255])
    np.testing.assert_array_equal(res[50, 50], [0, 0, 0])


def test_rounded_rectangle_corners_are_yx_not_xy(
    white_image: NDArray[np.uint8],
) -> None:
    # non-square, off-diagonal box so a (y, x) -> (x, y) swap moves the fill
    res = imgviz.draw.rounded_rectangle(
        white_image, (20, 40), (60, 95), radius=10, fill=(0, 0, 0)
    )

    # interior of the true box (rows 20-60, cols 40-95); white if axes swap
    np.testing.assert_array_equal(res[25, 80], [0, 0, 0])
    # interior of the swapped box only (rows 40-95, cols 20-60); black if axes swap
    np.testing.assert_array_equal(res[90, 30], [255, 255, 255])


def test_rounded_rectangle_in_place_mutates_pil_image() -> None:
    image = PIL.Image.new("RGB", (100, 100), (255, 255, 255))
    before = np.asarray(image).copy()

    imgviz.draw.rounded_rectangle_(image, (10, 10), (90, 90), radius=20, fill=(0, 0, 0))

    assert not np.array_equal(before, np.asarray(image))
