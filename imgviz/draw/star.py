import collections

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from .. import utils


def star(src, center, size, fill=None, outline=None):
    """Draw star on numpy array with Pillow.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    center: (2,) array-like
        center is (cy, cx).
    size: float
        Diameter to create the star.
    fill: int or (3,) array-like, optional
        RGB color to fill the mark. None for no fill. (default: None)
    outline: int or (3,) array-like, optional
        RGB color to draw the outline.

    Returns
    -------
    dst: numpy.ndarray
        Output image.

    """
    dst = utils.numpy_to_pillow(src)
    star_(
        img=dst,
        center=center,
        size=size,
        fill=fill,
        outline=outline,
    )
    return utils.pillow_to_numpy(dst)


def star_(img, center, size, fill=None, outline=None):
    if isinstance(fill, collections.abc.Iterable):
        fill = tuple(fill)
    if isinstance(outline, collections.abc.Iterable):
        outline = tuple(outline)

    draw = PIL.ImageDraw.Draw(img)

    radius = size / 2
    cy, cx = center

    # 5 mountains
    angles_m = np.arange(0, 5) * np.pi * 2 / 5 + (np.pi / 2)
    x_m = cx + radius * np.cos(angles_m)
    y_m = cy - radius * np.sin(angles_m)
    xy_m = np.stack((x_m, y_m), axis=1)

    # 5 valleys
    angles_v = angles_m + np.pi / 5
    length = radius / (np.sin(np.pi / 5) / np.tan(np.pi / 10) + np.cos(np.pi / 10))
    x_v = cx + length * np.cos(angles_v)
    y_v = cy - length * np.sin(angles_v)
    xy_v = np.stack((x_v, y_v), axis=1)

    xy = np.array(
        [
            xy_m[0],
            xy_v[0],
            xy_m[1],
            xy_v[1],
            xy_m[2],
            xy_v[2],
            xy_m[3],
            xy_v[3],
            xy_m[4],
            xy_v[4],
            xy_m[0],
        ]
    )
    xy = xy.flatten().tolist()
    draw.polygon(xy, fill=fill, outline=outline)
