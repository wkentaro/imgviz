from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def white_image() -> NDArray[np.uint8]:
    return np.full((200, 200, 3), 255, dtype=np.uint8)


def test_legend_preserves_shape_and_dtype(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.components.legend(
        white_image,
        items=[("cat", (255, 0, 0)), ("dog", (0, 0, 255))],
    )
    assert res.shape == white_image.shape
    assert res.dtype == white_image.dtype
    assert not np.array_equal(res, white_image)


def test_legend_wash_rounds_to_nearest() -> None:
    image = np.full((80, 120, 3), 200, dtype=np.uint8)
    res = imgviz.components.legend(
        image,
        items=[("a", (255, 0, 0)), ("bb", (0, 255, 0))],
        loc="lt",
    )
    # wash blends the region toward white at alpha=0.5:
    # 0.5 * 200 + 0.5 * 255 = 227.5, which must round to the nearest integer
    # (228), not truncate down to 227.
    assert res[5, 5].tolist() == [228, 228, 228]


def test_legend_empty_items_is_noop(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.components.legend(white_image, items=[])
    assert np.array_equal(res, white_image)


@pytest.mark.parametrize(
    ("loc", "corner"),
    [
        ("lt", (slice(0, 100), slice(0, 100))),
        ("rt", (slice(0, 100), slice(100, 200))),
        ("lb", (slice(100, 200), slice(0, 100))),
        ("rb", (slice(100, 200), slice(100, 200))),
    ],
)
def test_legend_loc_places_in_corner(
    white_image: NDArray[np.uint8],
    loc: Literal["lt", "rt", "lb", "rb"],
    corner: tuple[slice, slice],
) -> None:
    res = imgviz.components.legend(
        white_image,
        items=[("cat", (255, 0, 0))],
        loc=loc,
    )
    non_white = ~(res == 255).all(axis=2)
    assert non_white.any()
    assert non_white.sum() == non_white[corner].sum()


def test_legend_rejects_unsupported_loc(white_image: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match="unsupported loc"):
        imgviz.components.legend(
            white_image,
            items=[("cat", (255, 0, 0))],
            loc="bogus",  # type: ignore[arg-type]
        )
