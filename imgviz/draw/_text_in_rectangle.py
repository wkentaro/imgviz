import pathlib
from typing import Literal
from typing import NamedTuple
from typing import TypeAlias

import numpy as np
import PIL.Image
from numpy.typing import NDArray

from .. import _color
from .. import _utils
from ._rectangle import rectangle_
from ._text import text_
from ._text import text_size

_Loc: TypeAlias = Literal["lt", "rt", "lb", "rb", "lt+", "rt+", "lb-", "rb-"]


class _Aabb(NamedTuple):
    """Axis-Aligned Bounding Box."""

    y1: int
    x1: int
    y2: int
    x2: int


def text_in_rectangle_aabb(
    yx1: tuple[float, float] | NDArray[np.floating],
    yx2: tuple[float, float] | NDArray[np.floating],
    loc: _Loc,
    text: str,
    size: int,
    font_path: str | pathlib.Path | None = None,
) -> _Aabb:
    """Calculate bounding box for text in rectangle.

    Args:
        yx1: Coordinate of the rectangle minimum (y_min, x_min).
        yx2: Coordinate of the rectangle maximum (y_max, x_max).
        loc: Location of text. Must be one of: lt, rt, lb, rb, lt+, rt+, lb-,
            rb-.
        text: Text to draw.
        size: Text size in pixel.
        font_path: Font path.

    Returns:
        _Aabb named tuple with y1, x1, y2, x2 coordinates.
    """
    y1: int = int(round(yx1[0]))
    x1: int = int(round(yx1[1]))
    y2: int = int(round(yx2[0]))
    x2: int = int(round(yx2[1]))

    tsize: tuple[int, int] = text_size(text, size, font_path=font_path)

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
        raise ValueError(f"unsupported loc: {loc}")

    y1, x1 = yx
    y2, x2 = y1 + tsize[0] + 1, x1 + tsize[1] + 1

    return _Aabb(y1=y1, x1=x1, y2=y2, x2=x2)


def text_in_rectangle(
    image: NDArray[np.uint8],
    loc: _Loc,
    text: str,
    size: int,
    background: tuple[int, int, int] | NDArray[np.uint8],
    color: tuple[int, int, int] | NDArray[np.uint8] | None = None,
    yx1: tuple[float, float] | NDArray[np.floating] | None = None,
    yx2: tuple[float, float] | NDArray[np.floating] | None = None,
    font_path: str | pathlib.Path | None = None,
    keep_size: bool = False,
) -> NDArray[np.uint8]:
    """Draw text in a rectangle.

    Args:
        image: Input image.
        loc: Location of text. Must be one of: lt, rt, lb, rb, lt+, rt+, lb-,
            rb-.
        text: Text to draw.
        size: Text size in pixel.
        background: Background color in uint8.
        color: Text RGB color in uint8. If None, the color is determined by
            background color.
        yx1: Coordinate of the rectangle minimum (y_min, x_min). None for (0, 0).
        yx2: Coordinate of the rectangle maximum (y_max, x_max).
            (height-1, width-1).
        font_path: Font path.
        keep_size: Force to keep original size (size change happens with
            loc=xx+, xx-).

    Returns:
        Output image.
    """
    if color is None:
        color = _color.get_fg_color(background)

    height, width = image.shape[:2]
    y1, x1, y2, x2 = text_in_rectangle_aabb(
        yx1=(0, 0) if yx1 is None else yx1,
        yx2=(height - 1, width - 1) if yx2 is None else yx2,
        loc=loc,
        text=text,
        size=size,
        font_path=font_path,
    )

    dst = image

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

    dst = _utils.numpy_to_pillow(dst)
    rectangle_(
        image=dst,
        yx1=(y1, x1),
        yx2=(y2, x2),
        fill=background,
    )
    text_(
        image=dst,
        yx=(y1 + 1, x1 + 1),
        text=text,
        color=color,
        size=size,
        font_path=font_path,
    )
    return _utils.pillow_to_numpy(dst)


def text_in_rectangle_(
    image: PIL.Image.Image,
    loc: _Loc,
    text: str,
    size: int,
    background: tuple[int, int, int] | NDArray[np.uint8],
    color: tuple[int, int, int] | NDArray[np.uint8] | None = None,
    yx1: tuple[float, float] | NDArray[np.floating] | None = None,
    yx2: tuple[float, float] | NDArray[np.floating] | None = None,
    font_path: str | pathlib.Path | None = None,
) -> None:
    """Draw text in a rectangle on PIL image in-place.

    Args:
        image: PIL image to draw on (modified in-place).
        loc: Location of text. Must be one of: lt, rt, lb, rb, lt+, rt+, lb-,
            rb-.
        text: Text to draw.
        size: Text size in pixel.
        background: Background color in uint8.
        color: Text RGB color in uint8. If None, the color is determined by
            background color.
        yx1: Coordinate of the rectangle minimum (y_min, x_min). None for (0, 0).
        yx2: Coordinate of the rectangle maximum (y_max, x_max). None for
            (height-1, width-1).
        font_path: Font path.
    """
    if color is None:
        color = _color.get_fg_color(background)

    y1, x1, y2, x2 = text_in_rectangle_aabb(
        yx1=(0, 0) if yx1 is None else yx1,
        yx2=(image.height - 1, image.width - 1) if yx2 is None else yx2,
        loc=loc,
        text=text,
        size=size,
        font_path=font_path,
    )
    rectangle_(
        image=image,
        yx1=(y1, x1),
        yx2=(y2, x2),
        fill=background,
    )
    text_(
        image=image,
        yx=(y1 + 1, x1 + 1),
        text=text,
        color=color,
        size=size,
        font_path=font_path,
    )
