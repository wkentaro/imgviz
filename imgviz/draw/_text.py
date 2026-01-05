import pathlib

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from numpy.typing import NDArray

from .. import _utils
from ._ink import Ink
from ._ink import get_pil_ink

_here: pathlib.Path = pathlib.Path(__file__).parent
_default_font_path: pathlib.Path = _here / "fonts" / "DejaVuSansMono.ttf"


def _get_font(
    size: int, font_path: str | pathlib.Path | None = None
) -> PIL.ImageFont.FreeTypeFont:
    if font_path is None:
        font_path = _default_font_path
    font = PIL.ImageFont.truetype(font=str(font_path), size=size)
    return font


def text_size(
    text: str,
    size: int,
    font_path: str | pathlib.Path | None = None,
) -> tuple[int, int]:
    """Get text size (height and width).

    Args:
        text: Text.
        size: Pixel font size.
        font_path: Font path.

    Returns:
        Tuple of (height, width).
    """
    font = _get_font(size, font_path=font_path)

    text_width = 0
    text_height = 0
    for line in text.splitlines():
        if line == "":
            line = "\n"

        line_width, line_height = font.getbbox(line)[2:]
        text_width = max(text_width, line_width)
        text_height += line_height

    return text_height, text_width


def text(
    src: NDArray[np.uint8],
    yx: tuple[float, float],
    text: str,
    size: int,
    color: Ink = (0, 0, 0),
    font_path: str | pathlib.Path | None = None,
) -> NDArray[np.uint8]:
    """Draw text on numpy array with Pillow.

    Args:
        src: Input image.
        yx: Left top point of the text (y, x).
        text: Text to draw.
        size: Text size in pixel.
        color: Text RGB color in uint8. Default is (0, 0, 0), which is black.
        font_path: Font path. Default font is DejaVuSansMono.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(src)
    text_(image=dst, yx=yx, text=text, size=size, color=color, font_path=font_path)
    return _utils.pillow_to_numpy(dst)


def text_(
    image: PIL.Image.Image,
    yx: tuple[float, float],
    text: str,
    size: int,
    color: Ink = (0, 0, 0),
    font_path: str | pathlib.Path | None = None,
) -> None:
    """Draw text on PIL image in-place.

    Args:
        image: PIL image to draw on (modified in-place).
        yx: Left top point of the text (y, x).
        text: Text to draw.
        size: Text size in pixel.
        color: Text RGB color in uint8. Default is (0, 0, 0), which is black.
        font_path: Font path. Default font is DejaVuSansMono.
    """
    draw = PIL.ImageDraw.Draw(image)

    y1, x1 = yx
    font = _get_font(size=size, font_path=font_path)
    draw.text(xy=(x1, y1), text=text, fill=get_pil_ink(color), font=font)
