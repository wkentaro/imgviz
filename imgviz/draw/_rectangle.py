import numpy as np
import PIL.Image
import PIL.ImageDraw
from numpy.typing import NDArray

from .. import _utils
from ._ink import Ink
from ._ink import get_pil_ink


def rectangle(
    src: NDArray[np.uint8],
    yx1: tuple[float, float] | NDArray[np.floating],
    yx2: tuple[float, float] | NDArray[np.floating],
    fill: Ink | None = None,
    outline: Ink | None = None,
    width: int = 0,
) -> NDArray[np.uint8]:
    """Draw rectangle on numpy array with Pillow.

    Args:
        src: Input image.
        yx1: Minimum vertex (y_min, x_min) of the axis aligned bounding box.
        yx2: Maximum vertex (y_max, x_max) of the AABB.
        fill: RGB color to fill the mark. None for no fill.
        outline: RGB color to draw the outline.
        width: Rectangle line width.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(src)
    rectangle_(
        image=dst,
        yx1=yx1,
        yx2=yx2,
        fill=fill,
        outline=outline,
        width=width,
    )
    return _utils.pillow_to_numpy(dst)


def rectangle_(
    image: PIL.Image.Image,
    yx1: tuple[float, float] | NDArray[np.floating],
    yx2: tuple[float, float] | NDArray[np.floating],
    fill: Ink | None = None,
    outline: Ink | None = None,
    width: int = 0,
) -> None:
    """Draw rectangle on PIL image in-place.

    Args:
        image: PIL image to draw on (modified in-place).
        yx1: Minimum vertex (y_min, x_min) of the axis aligned bounding box.
        yx2: Maximum vertex (y_max, x_max) of the AABB.
        fill: RGB color to fill the mark. None for no fill.
        outline: RGB color to draw the outline.
        width: Rectangle line width.
    """
    draw = PIL.ImageDraw.Draw(image)

    y1, x1 = map(float, yx1)
    y2, x2 = map(float, yx2)
    draw.rectangle(
        xy=(x1, y1, x2, y2),
        fill=get_pil_ink(fill),
        outline=get_pil_ink(outline),
        width=width,
    )
