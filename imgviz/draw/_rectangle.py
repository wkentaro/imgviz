import collections

import PIL.Image
import PIL.ImageDraw

from .. import _utils


def rectangle(src, aabb1, aabb2, fill=None, outline=None, width=0):
    """Draw rectangle on numpy array with Pillow.

    Args:
        src: Input image.
        aabb1: Minimum vertex (y_min, x_min) of the axis aligned bounding box.
        aabb2: Maximum vertex (y_max, x_max) of the AABB.
        fill: RGB color to fill the mark. None for no fill.
        outline: RGB color to draw the outline.
        width: Rectangle line width.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(src)
    rectangle_(
        img=dst,
        aabb1=aabb1,
        aabb2=aabb2,
        fill=fill,
        outline=outline,
        width=width,
    )
    return _utils.pillow_to_numpy(dst)


def rectangle_(img, aabb1, aabb2, fill=None, outline=None, width=0):
    if isinstance(fill, collections.abc.Iterable):
        fill = tuple(fill)
    if isinstance(outline, collections.abc.Iterable):
        outline = tuple(outline)

    draw = PIL.ImageDraw.Draw(img)

    y1, x1 = aabb1
    y2, x2 = aabb2
    draw.rectangle(xy=(x1, y1, x2, y2), fill=fill, outline=outline, width=width)
