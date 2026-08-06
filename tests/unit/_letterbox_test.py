from typing import Literal
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz


def test_letterbox_landscape_to_square() -> None:
    img = np.random.uniform(0, 255, size=(15, 25, 3)).round().astype(np.uint8)

    dst = imgviz.letterbox(img, height=25, width=25, color=0)
    assert dst.shape == (25, 25, 3)
    assert dst.dtype == img.dtype
    assert (dst[:5] == 0).all()
    assert (dst[-5:] == 0).all()
    np.testing.assert_allclose(dst[5:-5], img)


def test_letterbox_landscape_to_landscape_smaller_height() -> None:
    img = np.random.uniform(0, 255, size=(15, 25, 3)).round().astype(np.uint8)

    dst = imgviz.letterbox(img, height=15, width=35, color=0)
    assert dst.shape == (15, 35, 3)
    assert dst.dtype == img.dtype
    assert (dst[:, :5] == 0).all()
    assert (dst[:, -5:] == 0).all()
    np.testing.assert_allclose(dst[:, 5:-5], img)


def test_letterbox_portrait_to_square() -> None:
    img = np.random.uniform(0, 255, size=(25, 15, 3)).round().astype(np.uint8)

    dst = imgviz.letterbox(img, height=25, width=25, color=0)
    assert dst.shape == (25, 25, 3)
    assert dst.dtype == img.dtype
    assert (dst[:, :5] == 0).all()
    assert (dst[:, -5:] == 0).all()
    np.testing.assert_allclose(dst[:, 5:-5], img)


@pytest.mark.parametrize("shape", [(100, 3, 3), (3, 100, 3)])
def test_letterbox_extreme_aspect_ratio_keeps_at_least_one_pixel(
    shape: tuple[int, int, int],
) -> None:
    img = np.zeros(shape, dtype=np.uint8)

    dst = imgviz.letterbox(img, height=1, width=1)  # a dimension would round to 0
    assert dst.shape == (1, 1, 3)
    assert dst.dtype == img.dtype


def test_letterbox_grayscale_hw() -> None:
    img = np.random.uniform(0, 255, size=(15, 25)).round().astype(np.uint8)

    dst = imgviz.letterbox(img, height=25, width=25, color=0)
    assert dst.shape == (25, 25)
    assert dst.dtype == img.dtype


def test_letterbox_rgba_hwca() -> None:
    img = np.random.uniform(0, 255, size=(15, 25, 4)).round().astype(np.uint8)

    dst = imgviz.letterbox(img, height=25, width=25, color=(0, 0, 0, 255))
    assert dst.shape == (25, 25, 4)
    assert dst.dtype == img.dtype
    assert (dst[:5, :, 3] == 255).all()
    assert (dst[-5:, :, 3] == 255).all()


def test_letterbox_color_tuple() -> None:
    img = np.zeros((10, 20, 3), dtype=np.uint8)

    dst = imgviz.letterbox(img, height=20, width=20, color=(255, 0, 0))
    assert dst.shape == (20, 20, 3)
    np.testing.assert_array_equal(dst[0, 0], [255, 0, 0])
    np.testing.assert_array_equal(dst[-1, -1], [255, 0, 0])
    np.testing.assert_array_equal(dst[10, 10], [0, 0, 0])


def test_letterbox_center_padding_within_one_pixel() -> None:
    img = np.full((10, 10, 3), 128, dtype=np.uint8)

    dst, mask = imgviz.letterbox(img, height=21, width=21, color=0, return_mask=True)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top_pad = int(rows[0])
    bottom_pad = int(mask.shape[0] - 1 - rows[-1])
    left_pad = int(cols[0])
    right_pad = int(mask.shape[1] - 1 - cols[-1])
    assert abs(top_pad - bottom_pad) <= 1
    assert abs(left_pad - right_pad) <= 1


def test_letterbox_return_mask_shape() -> None:
    img = np.random.uniform(0, 255, size=(10, 20, 3)).round().astype(np.uint8)

    dst, mask = imgviz.letterbox(img, height=30, width=30, color=0, return_mask=True)
    assert dst.shape == (30, 30, 3)
    assert mask.shape == (30, 30)
    assert mask.dtype == np.bool_
    assert mask.sum() > 0


def test_letterbox_identity_when_shape_matches() -> None:
    img = np.random.uniform(0, 255, size=(20, 20, 3)).round().astype(np.uint8)

    dst = imgviz.letterbox(img, height=20, width=20)
    np.testing.assert_array_equal(dst, img)


def test_letterbox_returns_copy_when_shape_matches() -> None:
    img = np.random.uniform(0, 255, size=(20, 20, 3)).round().astype(np.uint8)
    original = img.copy()

    dst = imgviz.letterbox(img, height=20, width=20)
    assert not np.shares_memory(dst, img)
    dst[...] = 0
    np.testing.assert_array_equal(img, original)


def test_letterbox_returns_copy_when_shape_matches_with_mask() -> None:
    img = np.random.uniform(0, 255, size=(20, 20, 3)).round().astype(np.uint8)
    original = img.copy()

    dst, mask = imgviz.letterbox(img, height=20, width=20, return_mask=True)
    assert not np.shares_memory(dst, img)
    dst[...] = 0
    np.testing.assert_array_equal(img, original)
    assert mask.shape == (20, 20)
    assert mask.all()


_Loc: TypeAlias = Literal["center", "lt", "rt", "lb", "rb"]


@pytest.mark.parametrize("loc", _Loc.__args__)
def test_letterbox_loc_square(loc: _Loc, show: bool) -> None:
    SHORT_SIZE: int = 10
    LONG_SIZE: int = 20
    shape: tuple[int, int, int] = (SHORT_SIZE, SHORT_SIZE, 3)

    image: NDArray[np.uint8] = (
        np.random.uniform(1, 255, size=shape).round().astype(np.uint8)
    )

    dst = imgviz.letterbox(image, height=LONG_SIZE, width=LONG_SIZE, color=0, loc=loc)
    if show:
        plt.imshow(dst)
        plt.show()

    assert dst.shape == (LONG_SIZE, LONG_SIZE, 3)
    assert (dst != 0).all()


@pytest.mark.parametrize("loc", _Loc.__args__)
def test_letterbox_loc_portrait(loc: _Loc, show: bool) -> None:
    SHORT_SIZE: int = 10
    LONG_SIZE: int = 20
    shape: tuple[int, int, int] = (LONG_SIZE, SHORT_SIZE, 3)

    image: NDArray[np.uint8] = (
        np.random.uniform(1, 255, size=shape).round().astype(np.uint8)
    )

    dst = imgviz.letterbox(image, height=LONG_SIZE, width=LONG_SIZE, color=0, loc=loc)
    if show:
        plt.imshow(dst)
        plt.show()

    assert dst.shape == (LONG_SIZE, LONG_SIZE, 3)
    size_diff: int = LONG_SIZE - SHORT_SIZE
    match loc:
        case "center":
            assert (dst[:, : size_diff // 2] == 0).all()
            assert (dst[:, -size_diff // 2 :] == 0).all()
        case "lt" | "lb":
            assert (dst[:, :size_diff] != 0).all()
            assert (dst[:, size_diff:] == 0).all()
        case "rt" | "rb":
            assert (dst[:, :-size_diff] == 0).all()
            assert (dst[:, -size_diff:] != 0).all()
        case _:
            raise ValueError(f"unknown loc: {loc}")


@pytest.mark.parametrize("loc", _Loc.__args__)
def test_letterbox_loc_landscape(loc: _Loc, show: bool) -> None:
    SHORT_SIZE: int = 10
    LONG_SIZE: int = 20
    shape: tuple[int, int, int] = (SHORT_SIZE, LONG_SIZE, 3)

    image: NDArray[np.uint8] = (
        np.random.uniform(1, 255, size=shape).round().astype(np.uint8)
    )

    dst = imgviz.letterbox(image, height=LONG_SIZE, width=LONG_SIZE, color=0, loc=loc)
    if show:
        plt.imshow(dst)
        plt.show()

    assert dst.shape == (LONG_SIZE, LONG_SIZE, 3)
    size_diff: int = LONG_SIZE - SHORT_SIZE
    match loc:
        case "center":
            assert (dst[: size_diff // 2, :] == 0).all()
            assert (dst[-size_diff // 2 :, :] == 0).all()
        case "lt" | "rt":
            assert (dst[:size_diff, :] != 0).all()
            assert (dst[size_diff:, :] == 0).all()
        case "lb" | "rb":
            assert (dst[:-size_diff, :] == 0).all()
            assert (dst[-size_diff:, :] != 0).all()
        case _:
            raise ValueError(f"unknown loc: {loc}")


def test_letterbox_rejects_unsupported_loc() -> None:
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="unsupported loc"):
        imgviz.letterbox(img, height=20, width=20, loc="bogus")  # type: ignore[arg-type]
