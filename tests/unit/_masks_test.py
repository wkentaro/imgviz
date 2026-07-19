import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz

_COLOR_RED: tuple[int, int, int] = (255, 0, 0)
_COLOR_WHITE: tuple[int, int, int] = (255, 255, 255)

_WHITE_IMAGE: NDArray[np.uint8] = np.full((100, 100, 3), _COLOR_WHITE, dtype=np.uint8)
_SIMPLE_MASK: NDArray[np.bool_] = np.zeros((100, 100), dtype=bool)
_SIMPLE_MASK[20:80, 20:80] = True


@pytest.mark.parametrize("fill", [_COLOR_RED, imgviz.fill.Solid(color=_COLOR_RED)])
@pytest.mark.parametrize("image", [None, _WHITE_IMAGE])
def test_mask2rgb_solid(
    show: bool,
    fill: imgviz.fill.Fill | tuple[int, int, int],
    image: NDArray[np.uint8] | None,
) -> None:
    mask: NDArray[np.bool_] = _SIMPLE_MASK

    result = imgviz.mask2rgb(mask=mask, image=image, fill=fill)
    if show:
        plt.imshow(result)
        plt.show()

    assert result.dtype == np.uint8
    assert result.shape == (100, 100, 3)
    if image is None:
        assert (result[mask] == _COLOR_RED).all()
        assert (result[~mask] == (0, 0, 0)).all()
    else:
        assert (result[mask] == (255, 128, 128)).all()
        assert (result[~mask] == _COLOR_WHITE).all()


@pytest.mark.parametrize("image", [None, _WHITE_IMAGE])
def test_mask2rgb_stripe(show: bool, image: NDArray[np.uint8] | None) -> None:
    mask: NDArray[np.bool_] = _SIMPLE_MASK

    result = imgviz.mask2rgb(
        mask=mask,
        image=image,
        fill=imgviz.fill.Stripe(color=_COLOR_RED, width=2, gap=2),
    )
    if show:
        plt.imshow(result)
        plt.show()

    assert result.dtype == np.uint8
    assert result.shape == (100, 100, 3)

    bg_color: tuple[int, int, int] = (0, 0, 0) if image is None else _COLOR_WHITE
    assert (result[~mask] == bg_color).all()

    stripe_color: tuple[int, int, int] = (
        _COLOR_RED if image is None else (255, 128, 128)
    )
    num_pixels: int = (result[mask] == stripe_color).all(axis=1).sum()
    assert 0.48 < num_pixels / mask.sum() < 0.52


def test_mask2rgb_stripe_angle_0_draws_horizontal_bands() -> None:
    result = imgviz.mask2rgb(
        mask=_SIMPLE_MASK,
        fill=imgviz.fill.Stripe(color=_COLOR_RED, angle=0.0, width=2, gap=2),
    )

    # bands run along rows, so each row is uniform; period 4 (width 2 + gap 2)
    # puts row 21 mid-stripe (21 % 4 == 1) and row 23 mid-gap (23 % 4 == 3).
    assert (result[21, 20:80] == _COLOR_RED).all()
    assert (result[23, 20:80] == (0, 0, 0)).all()


def test_mask2rgb_stripe_angle_half_pi_draws_vertical_bands() -> None:
    result = imgviz.mask2rgb(
        mask=_SIMPLE_MASK,
        fill=imgviz.fill.Stripe(color=_COLOR_RED, angle=np.pi / 2, width=2, gap=2),
    )

    # bands run along columns, so each column is uniform; col 21 is mid-stripe
    # and col 23 mid-gap (same period-4 phase as the horizontal case).
    assert (result[20:80, 21] == _COLOR_RED).all()
    assert (result[20:80, 23] == (0, 0, 0)).all()


def test_mask2rgb_does_not_mutate_rgba_image() -> None:
    image: NDArray[np.uint8] = imgviz.rgb2rgba(_WHITE_IMAGE)
    original = image.copy()

    imgviz.mask2rgb(mask=_SIMPLE_MASK, image=image, fill=_COLOR_RED)

    np.testing.assert_array_equal(image, original)


def test_mask2rgb_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="mask.ndim must be 2"):
        imgviz.mask2rgb(mask=np.zeros((2, 2, 1), dtype=bool))


def test_mask2rgb_rejects_non_bool() -> None:
    with pytest.raises(ValueError, match="mask.dtype must be bool"):
        imgviz.mask2rgb(mask=np.zeros((2, 2), dtype=np.uint8))


def test_stripe_validation() -> None:
    with pytest.raises(ValueError, match="width must be positive"):
        imgviz.fill.Stripe(color=_COLOR_RED, width=0)
    with pytest.raises(ValueError, match="width must be positive"):
        imgviz.fill.Stripe(color=_COLOR_RED, width=-1)
    with pytest.raises(ValueError, match="gap must be non-negative"):
        imgviz.fill.Stripe(color=_COLOR_RED, gap=-1)


def test_alpha_validation() -> None:
    with pytest.raises(ValueError, match="alpha must be in range"):
        imgviz.mask2rgb(mask=_SIMPLE_MASK, image=_WHITE_IMAGE, alpha=-0.1)
    with pytest.raises(ValueError, match="alpha must be in range"):
        imgviz.mask2rgb(mask=_SIMPLE_MASK, image=_WHITE_IMAGE, alpha=1.1)
