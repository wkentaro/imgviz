import numpy as np
import PIL.Image
import pytest
from numpy.typing import NDArray

import imgviz

GREEN = (0, 200, 0)
GRAY = (50, 50, 50)
RED = (255, 0, 0)


@pytest.fixture
def white_image() -> NDArray[np.uint8]:
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def test_progress_bar(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.progress_bar(
        white_image, yx1=(10, 10), yx2=(30, 90), value=0.5, fill=GREEN, background=GRAY
    )
    assert res.shape == white_image.shape
    assert res.dtype == white_image.dtype
    assert not np.array_equal(res, white_image)


def test_progress_bar_value_zero_is_background_only(
    white_image: NDArray[np.uint8],
) -> None:
    res = imgviz.draw.progress_bar(
        white_image, yx1=(10, 10), yx2=(30, 90), value=0.0, fill=GREEN, background=GRAY
    )
    assert (res[10:30, 10:90] == GRAY).all()


def test_progress_bar_value_one_is_fully_filled(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.progress_bar(
        white_image, yx1=(10, 10), yx2=(30, 90), value=1.0, fill=GREEN, background=GRAY
    )
    assert (res[10:30, 10:90] == GREEN).all()


def test_progress_bar_half(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.progress_bar(
        white_image, yx1=(10, 10), yx2=(30, 90), value=0.5, fill=GREEN, background=GRAY
    )
    assert tuple(res[20, 20]) == GREEN  # left half filled
    assert tuple(res[20, 80]) == GRAY  # right half is the track


@pytest.mark.parametrize(("value", "expected"), [(1.5, GREEN), (-1.0, GRAY)])
def test_progress_bar_clamps_value(
    white_image: NDArray[np.uint8], value: float, expected: tuple[int, int, int]
) -> None:
    res = imgviz.draw.progress_bar(
        white_image,
        yx1=(10, 10),
        yx2=(30, 90),
        value=value,
        fill=GREEN,
        background=GRAY,
    )
    assert (res[10:30, 10:90] == expected).all()


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_progress_bar_border(white_image: NDArray[np.uint8], value: float) -> None:
    res = imgviz.draw.progress_bar(
        white_image,
        yx1=(10, 10),
        yx2=(30, 90),
        value=value,
        fill=GREEN,
        background=GRAY,
        outline=RED,
    )
    assert tuple(res[10, 10]) == RED  # border drawn at every fill level


def test_progress_bar_border_does_not_overpaint_fill(
    white_image: NDArray[np.uint8],
) -> None:
    res = imgviz.draw.progress_bar(
        white_image,
        yx1=(10, 10),
        yx2=(30, 90),
        value=1.0,
        fill=GREEN,
        background=GRAY,
        outline=RED,
    )
    assert tuple(res[20, 50]) == GREEN  # 1px border leaves the interior filled


def test_progress_bar_in_place() -> None:
    pil = PIL.Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8))
    before = np.asarray(pil).copy()
    imgviz.draw.progress_bar_(
        pil, yx1=(10, 10), yx2=(30, 90), value=0.5, fill=GREEN, background=GRAY
    )
    assert not np.array_equal(np.asarray(pil), before)
