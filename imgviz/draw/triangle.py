import collections

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


def triangle(src, center, size, fill=None, outline=None):
    """Draw triangle on numpy array with Pillow.

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
    if isinstance(fill, collections.abc.Iterable):
        fill = tuple(fill)
    if isinstance(outline, collections.abc.Iterable):
        outline = tuple(outline)

    dst = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.Draw(dst)

    radius = size / 2
    cy, cx = center

    x = cx + radius * np.cos(np.deg2rad(np.arange(0, 3) * 120 + 90))
    y = cy - radius * np.sin(np.deg2rad(np.arange(0, 3) * 120 + 90))

    xy = np.stack((x, y), axis=1)
    xy = xy.flatten().tolist()
    draw.polygon(xy, fill=fill, outline=outline)

    return np.array(dst)
