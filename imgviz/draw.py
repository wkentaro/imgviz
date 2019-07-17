import os.path as osp

import matplotlib
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from . import color as color_module


def triangle(src, center, size, fill=None, outline=None):
    '''Draw triangle on numpy array with Pillow.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    center: (2,) array-like
        center is (cy, cx).
    size: float
        Diameter to create the star.
    fill: (3,) array-like, optional
        RGB color to fill the mark. None for no fill. (default: None)
    outline: (3,) array-like, optional
        RGB color to draw the outline.

    Returns
    -------
    dst: numpy.ndarray
        Output image.
    '''
    dst = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.Draw(dst)

    radius = size / 2
    cy, cx = center

    x = cx + radius * np.cos(np.deg2rad(np.arange(0, 3) * 120 + 90))
    y = cy - radius * np.sin(np.deg2rad(np.arange(0, 3) * 120 + 90))

    xy = np.stack((x, y), axis=1)
    xy = xy.flatten().tolist()
    draw.polygon(xy, fill=fill, outline=outline)

    return np.asarray(dst)


def star(src, center, size, fill=None, outline=None):
    '''Draw star on numpy array with Pillow.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    center: (2,) array-like
        center is (cy, cx).
    size: float
        Diameter to create the star.
    fill: (3,) array-like, optional
        RGB color to fill the mark. None for no fill. (default: None)
    outline: (3,) array-like, optional
        RGB color to draw the outline.

    Returns
    -------
    dst: numpy.ndarray
        Output image.
    '''
    dst = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.Draw(dst)

    radius = size / 2
    cy, cx = center

    # 5 mountains
    angles_m = np.arange(0, 5) * np.pi * 2 / 5 + (np.pi / 2)
    x_m = cx + radius * np.cos(angles_m)
    y_m = cy - radius * np.sin(angles_m)
    xy_m = np.stack((x_m, y_m), axis=1)

    # 5 valleys
    angles_v = angles_m + np.pi / 5
    l = radius * 1 / (
        np.sin(np.pi / 5) / np.tan(np.pi / 10) + np.cos(np.pi / 10)
    )
    x_v = cx + l * np.cos(angles_v)
    y_v = cy - l * np.sin(angles_v)
    xy_v = np.stack((x_v, y_v), axis=1)

    xy = np.array([
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
    ])
    xy = xy.flatten().tolist()
    draw.polygon(xy, fill=fill, outline=outline)

    return np.asarray(dst)


def circle(src, center, diameter, fill=None, outline=None, width=0):
    '''Draw circle on numpy array with Pillow.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    center: (2,) array-like
        center is (cy, cx).
    diameter: float
        Diameter of the circle.
    fill: (3,) array-like, optional
        RGB color to fill the mark. None for no fill. (default: None)
    outline: (3,) array-like, optional
        RGB color to draw the outline.
    width: int, optional
        Rectangle line width. (default: 0)

    Returns
    -------
    dst: numpy.ndarray
        Output image.
    '''
    dst = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.Draw(dst)

    cy, cx = center

    radius = diameter / 2.
    x1 = cx - radius
    x2 = x1 + diameter
    y1 = cy - radius
    y2 = y1 + diameter

    draw.ellipse([x1, y1, x2, y2], fill=fill, outline=outline, width=width)

    return np.asarray(dst)


def rectangle(src, aabb1, aabb2, fill=None, outline=None, width=0):
    '''Draw rectangle on numpy array with Pillow.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    aabb1, aabb2: (2,) array-like
        aabb1 is (y_min, x_min) and aabb2 is (y_max, x_max).
    fill: (3,) array-like, optional
        RGB color to fill the mark. None for no fill. (default: None)
    outline: (3,) array-like, optional
        RGB color to draw the outline.
    width: int, optional
        Rectangle line width. (default: 0)

    Returns
    -------
    dst: numpy.ndarray
        Output image.
    '''
    if outline is not None:
        outline = tuple(outline)
    if fill is not None:
        fill = tuple(fill)

    dst = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.ImageDraw(dst)

    y1, x1 = aabb1
    y2, x2 = aabb2
    draw.rectangle(
        xy=(x1, y1, x2, y2), fill=fill, outline=outline, width=width
    )

    return np.asarray(dst)


def _get_font(size):
    fonts_path = osp.join(
        osp.dirname(matplotlib.__file__), 'mpl-data/fonts/ttf'
    )
    font_path = osp.join(fonts_path, 'DejaVuSans.ttf')
    font = PIL.ImageFont.truetype(font=font_path, size=size)
    return font


def text_size(text, size):
    '''Get text size (height and width).

    Parameters
    ----------
    text: str
        Text.
    size: int
        Pixel font size.

    Returns
    -------
    height: int
        Text height.
    width: int
        Text width.
    '''
    font = _get_font(size)
    lines = text.splitlines()
    n_lines = len(lines)
    longest_line = max(lines, key=len)
    width, height = font.getsize(longest_line)
    return height * n_lines, width


def text(src, yx, text, size, color=(0, 0, 0)):
    '''Draw text on numpy array with Pillow.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    yx: (2,) array-like
        Left top point of the text.
    text: str
        Text to draw.
    size: int
        Text size in pixel.
    color: (3,) array-like
        Text RGB color in uint8.
        Default is (0, 0, 0), which is black.

    Returns
    -------
    dst: numpy.ndarray
        Output image.
    '''
    dst = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.ImageDraw(dst)

    y1, x1 = yx
    color = tuple(color)
    font = _get_font(size=size)
    draw.text(xy=(x1, y1), text=text, fill=color, font=font)

    return np.asarray(dst)


def text_in_rectangle(
    src,
    loc,
    text,
    size,
    background,
    color=None,
    aabb1=None,
    aabb2=None,
):
    '''Draw text in a rectangle.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    loc: str
        Location of text. It must be one of following: lt, rt, lb, or rb.
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

    Returns
    -------
    dst: numpy.ndarray
        Output image.
    '''
    if color is None:
        color = color_module.get_fg_color(background)

    height, width = src.shape[:2]

    y1, x1 = (0, 0) if aabb1 is None else aabb1
    y2, x2 = (height - 1, width - 1) if aabb2 is None else aabb2

    tsize = text_size(text, size)

    if loc == 'lt':
        yx = (y1, x1)
    elif loc == 'rt':
        yx = (y1, x2 - tsize[1] - 1)
    elif loc == 'lb':
        yx = (y2 - tsize[0] - 1, 0)
    elif loc == 'rb':
        yx = (y2 - tsize[0] - 1, x2 - tsize[1] - 1)
    else:
        raise ValueError('unsupported loc: {}'.format(loc))

    dst = globals()['rectangle'](
        src=src,
        aabb1=(yx[0], yx[1]),
        aabb2=(yx[0] + tsize[0] + 1, yx[1] + tsize[1] + 1),
        fill=background,
    )
    dst = globals()['text'](
        src=dst,
        yx=(yx[0] + 1, yx[1] + 1),
        text=text,
        color=color,
        size=size,
    )
    return dst
