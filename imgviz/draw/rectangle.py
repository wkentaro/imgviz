import collections

import PIL.Image

from .. import utils


def rectangle(src, aabb1, aabb2, fill=None, outline=None, width=0):
    """Draw rectangle on numpy array with Pillow.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    aabb1: array-like, (2,)
        Minimum vertex (y_min, x_min) of the axis aligned bounding box (AABB).
    aabb2: array-like, (2,)
        Maximum vertex (y_max, x_max) of the AABB.
    fill: int or array-like, (3,), optional
        RGB color to fill the mark. None for no fill. (default: None)
    outline: int or array-like, (3,), optional
        RGB color to draw the outline.
    width: int, optional
        Rectangle line width. (default: 0)

    Returns
    -------
    dst: numpy.ndarray
        Output image.

    """
    dst = utils.numpy_to_pillow(src)
    rectangle_(
        img=dst,
        aabb1=aabb1,
        aabb2=aabb2,
        fill=fill,
        outline=outline,
        width=width,
    )
    return utils.pillow_to_numpy(dst)


def rectangle_(img, aabb1, aabb2, fill=None, outline=None, width=0):
    if isinstance(fill, collections.abc.Iterable):
        fill = tuple(fill)
    if isinstance(outline, collections.abc.Iterable):
        outline = tuple(outline)

    draw = PIL.ImageDraw.ImageDraw(img)

    y1, x1 = aabb1
    y2, x2 = aabb2
    draw.rectangle(xy=(x1, y1, x2, y2), fill=fill, outline=outline, width=width)
