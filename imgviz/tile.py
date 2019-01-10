import math

import numpy as np

from .centerize import centerize
from .color import gray2rgb
from .color import rgb2rgba
from .draw import rectangle


def _tile(imgs, shape, dst):
    """Tile images which have same size.

    Parameters
    ----------
    imgs: numpy.ndarray
        Image list which should be tiled.
    shape: tuple of int
        Tile shape.
    dst:
        Image to put the tile on.
    """
    y_num, x_num = shape
    tile_w = imgs[0].shape[1]
    tile_h = imgs[0].shape[0]
    for y in range(y_num):
        for x in range(x_num):
            i = x + y * x_num
            if i < len(imgs):
                y1 = y * tile_h
                y2 = (y + 1) * tile_h
                x1 = x * tile_w
                x2 = (x + 1) * tile_w
                dst[y1:y2, x1:x2] = imgs[i]
    return dst


def _get_tile_shape(num):
    x_num = int(math.sqrt(num))  # floor
    y_num = 0
    while x_num * y_num < num:
        y_num += 1
    return x_num, y_num


def tile(
    imgs,
    shape=None,
    cval=None,
    border=None,
    border_width=None,
):
    assert isinstance(imgs, (list, tuple))
    imgs = imgs[:]

    if shape is None:
        shape = _get_tile_shape(len(imgs))

    if border:
        border = np.asarray(border, dtype=np.uint8)
        if border.ndim == 1:
            border = border[None].repeat(len(imgs), axis=0)
        assert border.ndim == 2
    else:
        border = (border,) * len(imgs)

    if border_width is None:
        border_width = 3

    # get max tile size to which each image should be resized
    max_h, max_w = np.array([img.shape[:2] for img in imgs]).max(axis=0)

    ndim = max(img.ndim for img in imgs)
    assert ndim in [2, 3]
    channel = max(img.shape[2] for img in imgs if img.ndim == 3)

    # tile images
    for i, img in enumerate(imgs):
        assert img.dtype == np.uint8

        if ndim == 3 and img.ndim == 2:
            img = gray2rgb(img)
        if channel == 4:
            img = rgb2rgba(img)

        img = centerize(src=img, shape=(max_h, max_w, channel), cval=cval)
        if border[i] is not None:
            img = rectangle(
                src=img,
                aabb1=(0, 0),
                aabb2=(img.shape[0] - 1, img.shape[1] - 1),
                color=border[i],
                width=border_width,
            )
        imgs[i] = img

    height = max_h * shape[0]
    width = max_w * shape[1]
    dst = np.zeros((height, width, channel), dtype=np.uint8)
    if cval:
        dst[:, :] = cval

    return _tile(imgs=imgs, shape=shape, dst=dst)
