import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz

_COLOR_RED: tuple[int, int, int] = (255, 0, 0)


@pytest.fixture
def image() -> NDArray[np.uint8]:
    return np.zeros((10, 10, 3), dtype=np.uint8)


@pytest.fixture
def mask() -> NDArray[np.bool_]:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    return mask


@pytest.fixture
def empty_mask() -> NDArray[np.bool_]:
    return np.zeros((10, 10), dtype=bool)


def test_solid_returns_new_array_when_copy_true(
    image: NDArray[np.uint8], mask: NDArray[np.bool_]
) -> None:
    original = image.copy()

    result = imgviz.fill.Solid(color=_COLOR_RED)(mask=mask, image=image, copy=True)

    assert not np.shares_memory(result, image)
    np.testing.assert_array_equal(image, original)
    assert (result[mask] == _COLOR_RED).all()


def test_solid_mutates_input_when_copy_false(
    image: NDArray[np.uint8], mask: NDArray[np.bool_]
) -> None:
    result = imgviz.fill.Solid(color=_COLOR_RED)(mask=mask, image=image, copy=False)

    assert result is image
    assert (image[mask] == _COLOR_RED).all()


def test_stripe_empty_mask_returns_copy_when_copy_true(
    image: NDArray[np.uint8], empty_mask: NDArray[np.bool_]
) -> None:
    result = imgviz.fill.Stripe(color=_COLOR_RED)(
        mask=empty_mask, image=image, copy=True
    )

    assert not np.shares_memory(result, image)
    np.testing.assert_array_equal(result, image)


def test_stripe_empty_mask_returns_input_when_copy_false(
    image: NDArray[np.uint8], empty_mask: NDArray[np.bool_]
) -> None:
    result = imgviz.fill.Stripe(color=_COLOR_RED)(
        mask=empty_mask, image=image, copy=False
    )

    assert result is image
