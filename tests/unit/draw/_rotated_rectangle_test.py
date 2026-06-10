import numpy as np
import PIL.Image
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def black_image() -> NDArray[np.uint8]:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _filled_bbox(image: NDArray[np.uint8]) -> tuple[int, int, int, int]:
    ys, xs = np.where(image[:, :, 0] > 0)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def test_rotated_rectangle(black_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.rotated_rectangle(
        black_image, center=(50, 50), size=(40, 20), angle=30, fill=(255, 255, 255)
    )
    assert res.shape == black_image.shape
    assert res.dtype == black_image.dtype
    assert not np.array_equal(res, black_image)


def test_rotated_rectangle_rejects_missing_fill_and_outline(
    black_image: NDArray[np.uint8],
) -> None:
    with pytest.raises(ValueError, match="at least one of `fill` or `outline`"):
        imgviz.draw.rotated_rectangle(
            black_image, center=(50, 50), size=(40, 20), angle=30
        )


def test_rotated_rectangle_zero_angle_is_axis_aligned(
    black_image: NDArray[np.uint8],
) -> None:
    res = imgviz.draw.rotated_rectangle(
        black_image, center=(50, 50), size=(40, 20), angle=0, fill=(255, 255, 255)
    )

    y1, y2, x1, x2 = _filled_bbox(res)
    # size=(height=40, width=20) centered at (50, 50)
    assert abs(y1 - 30) <= 1 and abs(y2 - 70) <= 1
    assert abs(x1 - 40) <= 1 and abs(x2 - 60) <= 1


def test_rotated_rectangle_ninety_degrees_swaps_extent(
    black_image: NDArray[np.uint8],
) -> None:
    res = imgviz.draw.rotated_rectangle(
        black_image, center=(50, 50), size=(40, 20), angle=90, fill=(255, 255, 255)
    )

    y1, y2, x1, x2 = _filled_bbox(res)
    # a 90-degree turn swaps the height and width extents
    assert abs((y2 - y1) - 20) <= 1
    assert abs((x2 - x1) - 40) <= 1


def test_rotated_rectangle_positive_angle_rotates_clockwise(
    black_image: NDArray[np.uint8],
) -> None:
    # a thin vertical bar; clockwise (image y-axis down) swings its top to the
    # right and its bottom to the left
    res = imgviz.draw.rotated_rectangle(
        black_image, center=(50, 50), size=(60, 6), angle=30, fill=(255, 255, 255)
    )

    ys, xs = np.where(res[:, :, 0] > 0)
    top_x_mean = xs[ys == ys.min()].mean()
    bottom_x_mean = xs[ys == ys.max()].mean()
    assert top_x_mean > 50
    assert bottom_x_mean < 50


def test_rotated_rectangle_outline_only(black_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.rotated_rectangle(
        black_image,
        center=(50, 50),
        size=(40, 20),
        angle=30,
        outline=(255, 255, 255),
        width=2,
    )

    assert not np.array_equal(res, black_image)
    # outline only leaves the interior unfilled
    np.testing.assert_array_equal(res[50, 50], [0, 0, 0])


def test_rotated_rectangle_preserves_area_under_rotation(
    black_image: NDArray[np.uint8],
) -> None:
    upright = imgviz.draw.rotated_rectangle(
        black_image, center=(50, 50), size=(40, 20), angle=0, fill=(255, 255, 255)
    )
    tilted = imgviz.draw.rotated_rectangle(
        black_image, center=(50, 50), size=(40, 20), angle=45, fill=(255, 255, 255)
    )

    area_upright = int((upright[:, :, 0] > 0).sum())
    area_tilted = int((tilted[:, :, 0] > 0).sum())
    assert not np.array_equal(upright, tilted)
    assert abs(area_upright - area_tilted) / area_upright < 0.05


def test_rotated_rectangle_in_place_mutates_pil_image() -> None:
    image = PIL.Image.new("RGB", (100, 100), (0, 0, 0))
    before = np.asarray(image).copy()

    imgviz.draw.rotated_rectangle_(
        image, center=(50, 50), size=(40, 20), angle=30, fill=(255, 255, 255)
    )

    assert not np.array_equal(before, np.asarray(image))
