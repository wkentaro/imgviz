from typing import Literal

import numpy as np
import PIL.Image
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def small_image() -> NDArray[np.uint8]:
    return np.full((20, 100, 3), 255, dtype=np.uint8)


def test_text_in_rectangle() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.text_in_rectangle(
        img,
        loc="lt",
        text="Hello",
        size=10,
        background=(0, 0, 0),
    )
    assert res.shape == img.shape
    assert res.dtype == np.uint8
    assert not np.array_equal(res, img)


@pytest.mark.parametrize("loc", ["lt", "lt+", "lb", "lb-"])
def test_text_in_rectangle_aabb_left_aligns_to_x1(
    loc: Literal["lt", "lt+", "lb", "lb-"],
) -> None:
    yx1 = (20, 50)
    yx2 = (80, 150)

    aabb = imgviz.draw.text_in_rectangle_aabb(
        yx1=yx1, yx2=yx2, loc=loc, text="Hi", size=10
    )

    assert aabb.x1 == yx1[1]


@pytest.mark.parametrize("loc", ["rt", "rt+", "rb", "rb-"])
def test_text_in_rectangle_aabb_right_aligns_to_x2(
    loc: Literal["rt", "rt+", "rb", "rb-"],
) -> None:
    yx1 = (20, 50)
    yx2 = (80, 150)

    aabb = imgviz.draw.text_in_rectangle_aabb(
        yx1=yx1, yx2=yx2, loc=loc, text="Hi", size=10
    )

    assert aabb.x2 == yx2[1] - 1  # text right edge sits one pixel inside x2


@pytest.mark.parametrize("loc", ["lt+", "lb-"])
def test_text_in_rectangle_grows_canvas_for_overflowing_loc(
    small_image: NDArray[np.uint8], loc: Literal["lt+", "lb-"]
) -> None:
    res = imgviz.draw.text_in_rectangle(
        small_image, loc=loc, text="Hi", size=10, background=(0, 0, 0)
    )

    assert res.shape[0] > small_image.shape[0]
    assert res.shape[1] == small_image.shape[1]
    assert res.dtype == np.uint8


def test_text_in_rectangle_keep_size_preserves_shape(
    small_image: NDArray[np.uint8],
) -> None:
    kept = imgviz.draw.text_in_rectangle(
        small_image, loc="lt+", text="Hi", size=10, background=(0, 0, 0), keep_size=True
    )
    grown = imgviz.draw.text_in_rectangle(
        small_image, loc="lt+", text="Hi", size=10, background=(0, 0, 0)
    )

    assert kept.shape == small_image.shape
    assert grown.shape[0] > small_image.shape[0]


def test_text_in_rectangle_auto_color_differs_from_explicit() -> None:
    img = np.full((40, 100, 3), 255, dtype=np.uint8)

    # The auto color for the (0, 0, 0) background arg is white; forcing black
    # text makes it invisible on the black rectangle, so the renders differ.
    auto = imgviz.draw.text_in_rectangle(
        img, loc="lt", text="Hi", size=10, background=(0, 0, 0)
    )
    explicit_black = imgviz.draw.text_in_rectangle(
        img, loc="lt", text="Hi", size=10, background=(0, 0, 0), color=(0, 0, 0)
    )

    assert not np.array_equal(auto, explicit_black)


def test_text_in_rectangle_draws_in_place() -> None:
    image = PIL.Image.fromarray(np.full((40, 100, 3), 255, dtype=np.uint8))
    before = np.array(image)

    imgviz.draw.text_in_rectangle_(
        image, loc="lt", text="Hi", size=10, background=(0, 0, 0)
    )

    assert not np.array_equal(np.array(image), before)
