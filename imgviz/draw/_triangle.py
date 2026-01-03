import collections

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from .. import _utils


def triangle(src, center, size, fill=None, outline=None):
    """Draw triangle on numpy array with Pillow.

    Args:
        src: Input image.
        center: Center point (cy, cx).
        size: Diameter to create the triangle.
        fill: RGB color to fill the mark. None for no fill.
        outline: RGB color to draw the outline.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(src)
    triangle_(img=dst, center=center, size=size, fill=fill, outline=outline)
    return _utils.pillow_to_numpy(dst)


def triangle_(img, center, size, fill=None, outline=None):
    if isinstance(fill, collections.abc.Iterable):
        fill = tuple(fill)
    if isinstance(outline, collections.abc.Iterable):
        outline = tuple(outline)

    draw = PIL.ImageDraw.Draw(img)

    radius = size / 2
    cy, cx = center

    x = cx + radius * np.cos(np.deg2rad(np.arange(0, 3) * 120 + 90))
    y = cy - radius * np.sin(np.deg2rad(np.arange(0, 3) * 120 + 90))

    xy = np.stack((x, y), axis=1)
    xy = xy.flatten().tolist()
    draw.polygon(xy, fill=fill, outline=outline)
