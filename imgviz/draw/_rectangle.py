import numpy as np
import PIL.Image
import PIL.ImageDraw
from numpy.typing import NDArray

from .. import _utils
from ._color import Color
from ._color import get_pil_color


def rectangle(
    src: NDArray[np.uint8],
    aabb1: tuple[float, float] | NDArray[np.floating],
    aabb2: tuple[float, float] | NDArray[np.floating],
    fill: Color | None = None,
    outline: Color | None = None,
    width: int = 0,
) -> NDArray[np.uint8]:
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


def rectangle_(
    img: PIL.Image.Image,
    aabb1: tuple[float, float] | NDArray[np.floating],
    aabb2: tuple[float, float] | NDArray[np.floating],
    fill: Color | None = None,
    outline: Color | None = None,
    width: int = 0,
) -> None:
    draw = PIL.ImageDraw.Draw(img)

    y1, x1 = map(float, aabb1)
    y2, x2 = map(float, aabb2)
    draw.rectangle(
        xy=(x1, y1, x2, y2),
        fill=get_pil_color(fill),
        outline=get_pil_color(outline),
        width=width,
    )
