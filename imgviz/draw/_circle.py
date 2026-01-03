import collections

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from .. import _utils


def circle(src, center, diameter, fill=None, outline=None, width=0):
    """Draw circle on numpy array with Pillow.

    Args:
        src: Input image.
        center: Center point (cy, cx).
        diameter: Diameter of the circle.
        fill: RGB color to fill the mark. None for no fill.
        outline: RGB color to draw the outline.
        width: Line width.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(src)
    circle_(
        img=dst,
        center=center,
        diameter=diameter,
        fill=fill,
        outline=outline,
        width=width,
    )
    return _utils.pillow_to_numpy(dst)


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
