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


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.0, GRAY), (0.0, GRAY), (1.0, GREEN), (1.5, GREEN)],
)
def test_progress_bar_fills_region_by_value(
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


def test_progress_bar_half(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.progress_bar(
        white_image, yx1=(10, 10), yx2=(30, 90), value=0.5, fill=GREEN, background=GRAY
    )
    assert tuple(res[20, 20]) == GREEN  # left half filled
    assert tuple(res[20, 80]) == GRAY  # right half is the track


@pytest.mark.parametrize(
    ("value", "green_col", "gray_col"),
    [(0.25, 29, 31), (0.5, 49, 51), (0.75, 69, 71)],
)
def test_progress_bar_fill_boundary_is_linear_in_value(
    white_image: NDArray[np.uint8], value: float, green_col: int, gray_col: int
) -> None:
    # The cases above only pin the endpoints; probing just inside and just
    # outside the boundary (at 10 + value * 80) catches any non-linear
    # value->pixel mapping, e.g. an accidental value**2 easing, that would
    # still leave the endpoints correct.
    res = imgviz.draw.progress_bar(
        white_image,
        yx1=(10, 10),
        yx2=(30, 90),
        value=value,
        fill=GREEN,
        background=GRAY,
    )
    assert tuple(res[20, green_col]) == GREEN
    assert tuple(res[20, gray_col]) == GRAY


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_progress_bar_rejects_non_finite_value(
    white_image: NDArray[np.uint8], value: float
) -> None:
    with pytest.raises(ValueError, match="value must be finite"):
        imgviz.draw.progress_bar(
            white_image,
            yx1=(10, 10),
            yx2=(30, 90),
            value=value,
            fill=GREEN,
            background=GRAY,
        )


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


def test_progress_bar_width(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.progress_bar(
        white_image,
        yx1=(10, 10),
        yx2=(30, 90),
        value=0.0,
        fill=GREEN,
        background=GRAY,
        outline=RED,
        width=3,
    )
    assert tuple(res[12, 12]) == RED  # 3px border reaches 2px into the box


def test_progress_bar_in_place() -> None:
    pil = PIL.Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8))
    before = np.asarray(pil).copy()
    imgviz.draw.progress_bar_(
        pil, yx1=(10, 10), yx2=(30, 90), value=0.5, fill=GREEN, background=GRAY
    )
    assert not np.array_equal(np.asarray(pil), before)
