import math

import numpy as np

from .centerize import centerize
from .color import gray2rgb
from .color import rgb2rgba


def _tile(imgs, shape, border=None, border_width=None):
    y_num, x_num = shape
    tile_h, tile_w, channel = imgs[0].shape

    if border is None:
        border_width = 0

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


def _get_tile_shape(num, hw_ratio=1):
    r_num = int(round(math.sqrt(num / hw_ratio)))  # weighted by wh_ratio
    c_num = 0
    while r_num * c_num < num:
        c_num += 1
    while (r_num - 1) * c_num >= num:
        r_num -= 1
    return r_num, c_num


def tile(
    imgs,
    shape=None,
    cval=None,
    border=None,
    border_width=None,
):
    """Tile images.

    Parameters
    ----------
    imgs: numpy.ndarray
        Image list which should be tiled.
    shape: tuple of int
        Tile shape.
    cval: array-like, optional
        Color to fill the background. Default is (0, 0, 0).
    border: array-like, optional
        Color for the border. If None, the border is not drawn.
    border_width: int
        Pixel size of the border.

    Returns
    -------
    dst: numpy.ndarray
        Tiled image.

    """
    imgs = list(imgs)  # copy

    # get max tile size to which each image should be resized
    max_h, max_w = np.array([img.shape[:2] for img in imgs]).max(axis=0)

    if shape is None:
        shape = _get_tile_shape(len(imgs), hw_ratio=1.0 * max_h / max_w)

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
    assert channel in [3, 4]

    # tile images
    for i in range(shape[0] * shape[1]):
        if i < len(imgs):
            img = imgs[i]
            assert img.dtype == np.uint8

            if ndim == 3 and img.ndim == 2:
                img = gray2rgb(img)
            if channel == 4 and img.shape[2] == 3:
                img = rgb2rgba(img)

            img = centerize(src=img, shape=(max_h, max_w, channel), cval=cval)
            imgs[i] = img
        else:
            img = np.full((max_h, max_w, channel), cval, dtype=np.uint8)
            imgs.append(img)

    return _tile(
        imgs=imgs, shape=shape, border=border, border_width=border_width
    )
