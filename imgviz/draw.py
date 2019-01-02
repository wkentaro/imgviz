import os.path as osp

import matplotlib
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


def rectangle(src, aabb1, aabb2, color, fill=None, width=0):
    '''Draw rectangle on numpy array with Pillow.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    aabb1, aabb2: (2,) array-like
        aabb1 is (y_min, x_min) and aabb2 is (y_max, x_max).
    color: (3,) array-like
        RGB color in uint8.
    fill: (3,) array-like, optional
        RGB color to fill the rectangle. None for no fill. (default: None)
    width: int, optional
        Rectangle line width. (default: 0)

    Returns
    -------
    dst: numpy.ndarray
        Output image.
    '''
    color = tuple(color)

    src_pil = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.ImageDraw(src_pil)

    y1, x1 = aabb1
    y2, x2 = aabb2
    draw.rectangle(xy=(x1, y1, x2, y2), fill=fill, outline=color, width=width)

    dst = np.asarray(src_pil)
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
    width, height = font.getsize(text)
    return height, width


def text(src, yx, text, color, size):
    '''Draw text on numpy array with Pillow.

    Parameters
    ----------
    src: numpy.ndarray
        Input image.
    yx: (2,) array-like
        Left top point of the text.
    text: str
        Text to draw.
    color: (3,) array-like
        Text RGB color in uint8
    size: int
        Text size in pixel.

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
    return dst
