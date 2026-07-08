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
        small_image, loc=loc, text="Hi", size=10, background=(255, 0, 0)
    )

    assert res.shape[0] > small_image.shape[0]
    assert res.shape[1] == small_image.shape[1]
    assert res.dtype == np.uint8
    # the grown canvas is filled with the background color, not garbage/zeros
    assert (res == [255, 0, 0]).all(axis=-1).any()


@pytest.mark.parametrize(("loc", "corner"), [("lt+", 0), ("lb-", -1)])
def test_text_in_rectangle_pads_with_full_background_color(
    small_image: NDArray[np.uint8], loc: Literal["lt+", "lb-"], corner: int
) -> None:
    background = (200, 50, 10)

    res = imgviz.draw.text_in_rectangle(
        small_image, loc=loc, text="Hi", size=10, background=background
    )

    # the grown row lies outside the narrow left-aligned rectangle at its right
    # edge, so it must carry the full background triplet, not background[0]
    # replicated across every channel.
    np.testing.assert_array_equal(res[corner, -1], background)


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


def test_text_in_rectangle_auto_color_is_white_on_black_background() -> None:
    img = np.full((40, 100, 3), 255, dtype=np.uint8)

    # with no explicit color, a black background auto-selects white text
    auto = imgviz.draw.text_in_rectangle(
        img, loc="lt", text="Hi", size=10, background=(0, 0, 0)
    )
    explicit_white = imgviz.draw.text_in_rectangle(
        img, loc="lt", text="Hi", size=10, background=(0, 0, 0), color=(255, 255, 255)
    )

    np.testing.assert_array_equal(auto, explicit_white)


def test_text_in_rectangle_draws_in_place() -> None:
    image = PIL.Image.fromarray(np.full((40, 100, 3), 255, dtype=np.uint8))
    before = np.array(image)

    imgviz.draw.text_in_rectangle_(
        image, loc="lt", text="Hi", size=10, background=(0, 0, 0)
    )

    assert not np.array_equal(np.array(image), before)


def test_text_in_rectangle_rejects_unsupported_loc(
    small_image: NDArray[np.uint8],
) -> None:
    with pytest.raises(ValueError, match="unsupported loc"):
        imgviz.draw.text_in_rectangle(
            small_image,
            loc="bogus",  # type: ignore[arg-type]
            text="Hi",
            size=10,
            background=(0, 0, 0),
        )
