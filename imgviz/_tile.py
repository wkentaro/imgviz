from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from ._centerize import centerize
from ._color import gray2rgb
from ._color import rgb2rgba
from .draw import Ink


def _tile(
    imgs: list[NDArray[np.uint8]],
    shape: tuple[int, int],
    border: NDArray[np.uint8] | None = None,
    border_width: int | None = None,
) -> NDArray[np.uint8]:
    y_num, x_num = shape
    tile_h, tile_w, channel = imgs[0].shape

    if border is None:
        border_width = 0
    if border_width is None:
        raise ValueError("border_width must be provided when border is not None")

    dst = np.zeros(
        (
            tile_h * y_num + border_width * (y_num - 1),
            tile_w * x_num + border_width * (x_num - 1),
            channel,
        ),
        dtype=np.uint8,
    )
    if border is not None:
        dst[...] = border

    for y in range(y_num):
        for x in range(x_num):
            i = x + y * x_num
            if i < len(imgs):
                y1 = y * tile_h + y * border_width
                y2 = y1 + tile_h
                x1 = x * tile_w + x * border_width
                x2 = x1 + tile_w
                dst[y1:y2, x1:x2] = imgs[i]
    return dst


def _get_tile_shape(num: int, hw_ratio: float = 1) -> tuple[int, int]:
    r_num = int(round(math.sqrt(num / hw_ratio)))  # weighted by wh_ratio
    c_num = 0
    while r_num * c_num < num:
        c_num += 1
    while (r_num - 1) * c_num >= num:
        r_num -= 1
    return r_num, c_num


def tile(
    imgs: Iterable[NDArray],
    row: int | None = None,
    col: int | None = None,
    cval: Ink | None = None,
    border: Ink | None = None,
    border_width: int | None = None,
) -> NDArray[np.uint8]:
    """Tile images.

    Args:
        imgs: Image list which should be tiled.
        row: Number of rows in the tile grid. If None, auto-calculated.
        col: Number of columns in the tile grid. If None, auto-calculated.
        cval: Color to fill the background.
        border: Color for the border. If None, the border is not drawn.
        border_width: Pixel size of the border.

    Returns:
        Tiled image.
    """
    imgs = list(imgs)  # copy

    # get max tile size to which each image should be resized
    max_h, max_w = np.array([img.shape[:2] for img in imgs]).max(axis=0)

    if row is None and col is None:
        shape = _get_tile_shape(len(imgs), hw_ratio=1.0 * max_h / max_w)
    elif row is None:
        if not isinstance(col, int):
            raise TypeError(f"col must be int, but got {type(col).__name__}")
        if col <= 0:
            raise ValueError(f"col must be positive, but got {col}")
        shape = (math.ceil(len(imgs) / col), col)
    elif col is None:
        if not isinstance(row, int):
            raise TypeError(f"row must be int, but got {type(row).__name__}")
        if row <= 0:
            raise ValueError(f"row must be positive, but got {row}")
        shape = (row, math.ceil(len(imgs) / row))
    else:
        if not isinstance(row, int):
            raise TypeError(f"row must be int, but got {type(row).__name__}")
        if not isinstance(col, int):
            raise TypeError(f"col must be int, but got {type(col).__name__}")
        if row <= 0 or col <= 0:
            raise ValueError(
                f"row and col must be positive, but got row={row}, col={col}"
            )
        shape = (row, col)

    imgs = imgs[: shape[0] * shape[1]]

    if cval is None:
        cval = 0

    if border is not None:
        border = np.asarray(border, dtype=np.uint8)

    if border_width is None:
        border_width = 3

    ndim = max(img.ndim for img in imgs)
    if ndim == 3:
        channel = max(img.shape[2] for img in imgs if img.ndim == 3)
    else:
        ndim = 3  # gray images will be converted to rgb
        channel = 3  # all gray
    if channel not in [3, 4]:
        raise ValueError(f"channel must be 3 or 4, but got {channel}")

    # tile images
    for i in range(shape[0] * shape[1]):
        img: NDArray
        if i < len(imgs):
            img = imgs[i]
            if img.dtype != np.uint8:
                raise ValueError(f"img dtype must be np.uint8, but got {img.dtype}")

            if ndim == 3 and img.ndim == 2:
                img = gray2rgb(img)
            if channel == 4 and img.shape[2] == 3:
                img = rgb2rgba(img)

            img = centerize(src=img, height=max_h, width=max_w, cval=cval)
            imgs[i] = img
        else:
            img = np.full((max_h, max_w, channel), cval, dtype=np.uint8)
            imgs.append(img)

    return _tile(imgs=imgs, shape=shape, border=border, border_width=border_width)
