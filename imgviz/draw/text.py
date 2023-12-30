import os.path as osp

import matplotlib
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from .. import utils


def _get_font(size, font_path=None):
    if font_path is None:
        fonts_path = osp.join(osp.dirname(matplotlib.__file__), "mpl-data/fonts/ttf")
        font_path = osp.join(fonts_path, "DejaVuSansMono.ttf")
    font = PIL.ImageFont.truetype(font=font_path, size=size)
    return font


def text_size(text, size, font_path=None):
    """Get text size (height and width).

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

    """
    font = _get_font(size, font_path=font_path)

    text_width = 0
    text_height = 0
    for line in text.splitlines():
        if line == "":
            line = "\n"

        if hasattr(font, "getbbox"):
            line_width, line_height = font.getbbox(line)[2:]
        else:
            line_width, line_height = font.getsize(line)
        text_width = max(text_width, line_width)
        text_height += line_height

    return text_height, text_width


def text(src, yx, text, size, color=(0, 0, 0), font_path=None):
    """Draw text on numpy array with Pillow.

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
    font_path: str
        Default font is DejaVuSansMono in matplotlib.

    Returns
    -------
    dst: numpy.ndarray
        Output image.

    """
    dst = utils.numpy_to_pillow(src)
    text_(img=dst, yx=yx, text=text, size=size, color=color, font_path=font_path)
    return utils.pillow_to_numpy(dst)


def text_(img, yx, text, size, color=(0, 0, 0), font_path=None):
    draw = PIL.ImageDraw.ImageDraw(img)

    y1, x1 = yx
    color = tuple(color)
    font = _get_font(size=size, font_path=font_path)
    draw.text(xy=(x1, y1), text=text, fill=color, font=font)
