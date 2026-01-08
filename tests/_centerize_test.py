from typing import Literal
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz


def test_centerize():
    img = np.random.uniform(0, 255, size=(15, 25, 3)).round().astype(np.uint8)

    dst = imgviz.centerize(img, height=25, width=25, cval=0)
    assert dst.shape == (25, 25, 3)
    assert dst.dtype == img.dtype
    assert (dst[:5] == 0).all()
    assert (dst[-5:] == 0).all()
    np.testing.assert_allclose(dst[5:-5], img)

    dst = imgviz.centerize(img, height=15, width=35, cval=0)
    assert dst.shape == (15, 35, 3)
    assert dst.dtype == img.dtype
    assert (dst[:, :5] == 0).all()
    assert (dst[:, -5:] == 0).all()
    np.testing.assert_allclose(dst[:, 5:-5], img)


_Loc: TypeAlias = Literal["center", "lt", "rt", "lb", "rb"]


@pytest.mark.parametrize("loc", _Loc.__args__)
def test_centerize_loc_square(loc: _Loc, show: bool) -> None:
    SHORT_SIZE: int = 10
    LONG_SIZE: int = 20
    shape: tuple[int, int, int] = (SHORT_SIZE, SHORT_SIZE, 3)

    image: NDArray[np.uint8] = (
        np.random.uniform(1, 255, size=shape).round().astype(np.uint8)
    )

    dst = imgviz.centerize(image, height=LONG_SIZE, width=LONG_SIZE, cval=0, loc=loc)
    if show:
        plt.imshow(dst)
        plt.show()

    assert dst.shape == (LONG_SIZE, LONG_SIZE, 3)
    assert (dst != 0).all()


@pytest.mark.parametrize("loc", _Loc.__args__)
def test_centerize_loc_portrait(loc: _Loc, show: bool) -> None:
    SHORT_SIZE: int = 10
    LONG_SIZE: int = 20
    shape: tuple[int, int, int] = (LONG_SIZE, SHORT_SIZE, 3)

    image: NDArray[np.uint8] = (
        np.random.uniform(1, 255, size=shape).round().astype(np.uint8)
    )

    dst = imgviz.centerize(image, height=LONG_SIZE, width=LONG_SIZE, cval=0, loc=loc)
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
def test_centerize_loc_landscape(loc: _Loc, show: bool) -> None:
    SHORT_SIZE: int = 10
    LONG_SIZE: int = 20
    shape: tuple[int, int, int] = (SHORT_SIZE, LONG_SIZE, 3)

    image: NDArray[np.uint8] = (
        np.random.uniform(1, 255, size=shape).round().astype(np.uint8)
    )

    dst = imgviz.centerize(image, height=LONG_SIZE, width=LONG_SIZE, cval=0, loc=loc)
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
