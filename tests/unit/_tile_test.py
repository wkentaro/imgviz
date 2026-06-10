import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz


def test_tile() -> None:
    img1 = np.random.uniform(0, 255, (15, 25, 3)).round().astype(np.uint8)
    img2 = np.random.uniform(0, 255, (25, 25, 3)).round().astype(np.uint8)
    img3 = np.random.uniform(0, 255, (25, 10, 3)).round().astype(np.uint8)
    tiled = imgviz.tile([img1, img2, img3], row=1, col=3)

    assert tiled.shape == (25, 75, 3)
    assert tiled.dtype == np.uint8


@pytest.fixture
def images() -> list[NDArray[np.uint8]]:
    return [np.full((10, 12, 3), i * 10, dtype=np.uint8) for i in range(4)]


def test_tile_auto_shape(images: list[NDArray[np.uint8]]) -> None:
    tiled = imgviz.tile(images)

    assert tiled.shape == (20, 24, 3)
    assert tiled.dtype == np.uint8


def test_tile_row_only(images: list[NDArray[np.uint8]]) -> None:
    tiled = imgviz.tile(images, row=1)

    assert tiled.shape == (10, 48, 3)


def test_tile_col_only(images: list[NDArray[np.uint8]]) -> None:
    tiled = imgviz.tile(images, col=1)

    assert tiled.shape == (40, 12, 3)


def test_tile_draws_border(images: list[NDArray[np.uint8]]) -> None:
    tiled = imgviz.tile(images, row=2, col=2, border=(255, 0, 0), border_width=3)

    assert tiled.shape == (23, 27, 3)
    assert (tiled[10:13] == (255, 0, 0)).all()
    assert (tiled[:, 12:15] == (255, 0, 0)).all()
    assert (tiled[0:10, 0:12] == 0).all()  # first image cell is left intact


def test_tile_promotes_gray_to_rgb() -> None:
    gray = [np.full((10, 12), 5, dtype=np.uint8) for _ in range(3)]

    tiled = imgviz.tile(gray, row=1, col=3)

    assert tiled.shape == (10, 36, 3)
    assert tiled.dtype == np.uint8
    assert (tiled == 5).all()  # gray value is broadcast across rgb channels


def test_tile_promotes_rgb_to_rgba() -> None:
    rgba = np.full((10, 12, 4), 5, dtype=np.uint8)
    rgb = np.full((10, 12, 3), 5, dtype=np.uint8)

    tiled = imgviz.tile([rgba, rgb], row=1, col=2)

    assert tiled.shape == (10, 24, 4)
    assert (tiled[:, :12, 3] == 5).all()  # already-rgba tile keeps its alpha
    assert (tiled[:, 12:, 3] == 255).all()  # rgb tile gains an opaque alpha


def test_tile_pads_short_grid() -> None:
    tiled = imgviz.tile([np.full((10, 12, 3), 7, dtype=np.uint8)], row=1, col=2, cval=0)

    assert tiled.shape == (10, 24, 3)
    assert (tiled[:, :12] == 7).all()  # image cell is written faithfully
    assert (tiled[:, 12:] == 0).all()


@pytest.mark.parametrize(("row", "col"), [(0, None), (None, 0), (-1, None), (None, -1)])
def test_tile_rejects_non_positive_row_col(
    images: list[NDArray[np.uint8]], row: int | None, col: int | None
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        imgviz.tile(images, row=row, col=col)


@pytest.mark.parametrize("kwargs", [{"col": "2"}, {"row": "2"}])
def test_tile_rejects_non_int_dimension(
    images: list[NDArray[np.uint8]], kwargs: dict[str, str]
) -> None:
    with pytest.raises(TypeError, match="must be int"):
        imgviz.tile(images, **kwargs)  # type: ignore[arg-type]


def test_tile_rejects_non_uint8() -> None:
    with pytest.raises(ValueError, match="image dtype must be np.uint8"):
        imgviz.tile([np.zeros((4, 4, 3), dtype=np.float32)], row=1, col=1)
