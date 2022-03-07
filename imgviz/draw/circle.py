import collections

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from .. import utils


def circle(src, center, diameter, fill=None, outline=None, width=0):
    """Draw circle on numpy array with Pillow.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    center: (2,) array-like
        center is (cy, cx).
    diameter: float
        Diameter of the circle.
    fill: int or (3,) array-like, optional
        RGB color to fill the mark. None for no fill. (default: None)
    outline: int or (3,) array-like, optional
        RGB color to draw the outline.
    width: int, optional
        Rectangle line width. (default: 0)

    Returns
    -------
    dst: numpy.ndarray
        Output image.

    """
    dst = utils.numpy_to_pillow(src)
    circle_(
        img=dst,
        center=center,
        diameter=diameter,
        fill=fill,
        outline=outline,
        width=width,
    )
    return utils.pillow_to_numpy(dst)


def circle_(img, center, diameter, fill=None, outline=None, width=0):
    if isinstance(fill, collections.abc.Iterable):
        fill = tuple(fill)
    if isinstance(outline, collections.abc.Iterable):
        outline = tuple(outline)

    draw = PIL.ImageDraw.Draw(img)

    cy, cx = center

    radius = diameter / 2.0
    x1 = cx - radius
    x2 = x1 + diameter
    y1 = cy - radius
    y2 = y1 + diameter

    draw.ellipse([x1, y1, x2, y2], fill=fill, outline=outline, width=width)
