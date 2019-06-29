import os.path as osp

import matplotlib
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


def circle(src, center, diameter, fill=None, outline=None, width=0):
    '''Draw circle on numpy array with Pillow.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    center: (2,) array-like
        center is (cy, cx).
    fill: (3,) array-like, optional
        RGB color to fill the circle. None for no fill. (default: None)
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

    radius = diameter / 2
    x1 = cx - radius
    x2 = x1 + diameter
    y1 = cy - radius
    y2 = y1 + diameter

    draw.ellipse([x1, y1, x2, y2], fill=fill, outline=outline, width=width)

    dst = np.asarray(dst)
    dst.setflags(write=True)
    return dst


def rectangle(src, aabb1, aabb2, fill=None, outline=None, width=0):
    '''Draw rectangle on numpy array with Pillow.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    aabb1, aabb2: (2,) array-like
        aabb1 is (y_min, x_min) and aabb2 is (y_max, x_max).
    fill: (3,) array-like, optional
        RGB color to fill the rectangle. None for no fill. (default: None)
    outline: (3,) array-like
        RGB color of the outline in uint8.
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

    src_pil = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.ImageDraw(src_pil)

    y1, x1 = aabb1
    y2, x2 = aabb2
    draw.rectangle(
        xy=(x1, y1, x2, y2), fill=fill, outline=outline, width=width
    )

    dst = np.asarray(src_pil)
    dst.setflags(write=True)
    return dst


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
    src_pil = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.ImageDraw(src_pil)

    y1, x1 = yx
    color = tuple(color)
    font = _get_font(size=size)
    draw.text(xy=(x1, y1), text=text, fill=color, font=font)

    dst = np.asarray(src_pil)
    dst.setflags(write=True)
    return dst


def text_in_rectangle(src, loc, text, size, background, color=(0, 0, 0)):
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
        Default is (0, 0, 0), which is black.

    Returns
    -------
    dst: numpy.ndarray
        Output image.
    '''

    tsize = text_size(text, size)

    height, width = src.shape[:2]
    if loc == 'lt':
        yx = (0, 0)
    elif loc == 'rt':
        yx = (0, (width - 1) - tsize[1] - 1)
    elif loc == 'lb':
        yx = ((height - 1) - tsize[0] - 1, 0)
    elif loc == 'rb':
        yx = ((height - 1) - tsize[0] - 1, (width - 1) - tsize[1] - 1)
    else:
        raise ValueError('unsupported loc: {}'.format(loc))

    dst = rectangle(
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
