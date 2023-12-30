import numpy as np

from .. import color as color_module
from .. import utils
from .rectangle import rectangle_
from .text import text_
from .text import text_size


def text_in_rectangle_aabb(img_shape, loc, text, size, aabb1, aabb2, font_path=None):
    height, width = img_shape[:2]

    y1, x1 = (0, 0) if aabb1 is None else aabb1
    y2, x2 = (height - 1, width - 1) if aabb2 is None else aabb2

    tsize = text_size(text, size, font_path=font_path)

    if loc == "lt":
        yx = (y1, x1)
    elif loc == "lt+":
        yx = (y1 - tsize[0] - 2, x1)
    elif loc == "rt":
        yx = (y1, x2 - tsize[1] - 2)
    elif loc == "rt+":
        yx = (y1 - tsize[0] - 2, x2 - tsize[1] - 2)
    elif loc == "lb":
        yx = (y2 - tsize[0] - 2, 0)
    elif loc == "lb-":
        yx = (y2, 0)
    elif loc == "rb":
        yx = (y2 - tsize[0] - 2, x2 - tsize[1] - 2)
    elif loc == "rb-":
        yx = (y2, x2 - tsize[1] - 2)
    else:
        raise ValueError("unsupported loc: {}".format(loc))

    y1, x1 = yx
    y2, x2 = y1 + tsize[0] + 1, x1 + tsize[1] + 1

    return np.array([y1, x1, y2, x2])


def text_in_rectangle(
    src,
    loc,
    text,
    size,
    background,
    color=None,
    aabb1=None,
    aabb2=None,
    font_path=None,
    keep_size=False,
):
    """Draw text in a rectangle.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    loc: str
        Location of text. It must be one of following:
        lt, rt, lb, rb, lt+, rt+, lb-, rb-.
    text: str
        Text to draw.
    size: int
        Text size in pixel.
    background: (3,) array-like
        Background color in uint8.
    color: (3,) array-like
        Text RGB color in uint8.
        If None, the color is determined by background color.
        (default: None)
    aabb1, aabb2: (2,) array-like
        Coordinate of the rectangle (y_min, x_min), (y_max, x_max).
        Default is (0, 0), (height, width).
    keep_size: bool
        Force to keep original size (size change happens with loc=xx+, xx-).

    Returns
    -------
    dst: numpy.ndarray
        Output image.

    """
    if color is None:
        color = color_module.get_fg_color(background)

    height, width = src.shape[:2]
    y1, x1, y2, x2 = text_in_rectangle_aabb(
        img_shape=src.shape,
        loc=loc,
        text=text,
        size=size,
        aabb1=aabb1,
        aabb2=aabb2,
        font_path=font_path,
    )

    dst = src

    if not keep_size:
        constant_values = (
            (background[0],),
            (background[1],),
            (background[2],),
        )
        if y1 < 0:
            pad = -y1
            dst = np.pad(
                dst,
                ((pad, 0), (0, 0), (0, 0)),
                mode="constant",
                constant_values=constant_values,
            )
            y1 += pad
            y2 += pad
        if y2 > height:
            pad = y2 - height
            dst = np.pad(
                dst,
                ((0, pad), (0, 0), (0, 0)),
                mode="constant",
                constant_values=constant_values,
            )

    dst = utils.numpy_to_pillow(dst)
    rectangle_(
        img=dst,
        aabb1=(y1, x1),
        aabb2=(y2, x2),
        fill=background,
    )
    text_(
        img=dst,
        yx=(y1 + 1, x1 + 1),
        text=text,
        color=color,
        size=size,
        font_path=font_path,
    )
    return utils.pillow_to_numpy(dst)


def text_in_rectangle_(
    img,
    loc,
    text,
    size,
    background,
    color=None,
    aabb1=None,
    aabb2=None,
    font_path=None,
):
    if color is None:
        color = color_module.get_fg_color(background)

    y1, x1, y2, x2 = text_in_rectangle_aabb(
        img_shape=(img.height, img.width),
        loc=loc,
        text=text,
        size=size,
        aabb1=aabb1,
        aabb2=aabb2,
        font_path=font_path,
    )
    rectangle_(
        img=img,
        aabb1=(y1, x1),
        aabb2=(y2, x2),
        fill=background,
    )
    text_(
        img=img,
        yx=(y1 + 1, x1 + 1),
        text=text,
        color=color,
        size=size,
        font_path=font_path,
    )
