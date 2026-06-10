from typing import Literal

import numpy as np
import pytest

import imgviz


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
