import numpy as np
import PIL.Image
import PIL.ImageDraw
from numpy.typing import NDArray

from .. import _utils
from ._ink import Ink
from ._ink import get_pil_ink


def box_corners(
    image: NDArray[np.uint8],
    yx1: tuple[float, float] | NDArray[np.floating],
    yx2: tuple[float, float] | NDArray[np.floating],
    fill: Ink,
    length: int = 12,
    width: int = 2,
) -> NDArray[np.uint8]:
    """Draw the four L-shaped corners of a rectangle on numpy array with Pillow.

    Args:
        image: Input image.
        yx1: Minimum vertex (y_min, x_min) of the axis aligned bounding box.
        yx2: Maximum vertex (y_max, x_max) of the AABB.
        fill: RGB color to draw the corners.
        length: Corner tick length in pixels. Clamped per axis to the box
            width/height, so a large value degrades to a full rectangle.
        width: Line width.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(image)
    box_corners_(
        image=dst,
        yx1=yx1,
        yx2=yx2,
        fill=fill,
        length=length,
        width=width,
    )
    return _utils.pillow_to_numpy(dst)


def box_corners_(
    image: PIL.Image.Image,
    yx1: tuple[float, float] | NDArray[np.floating],
    yx2: tuple[float, float] | NDArray[np.floating],
    fill: Ink,
    length: int = 12,
    width: int = 2,
) -> None:
    """Draw the four L-shaped corners of a rectangle on PIL image in-place.

    Args:
        image: PIL image to draw on (modified in-place).
        yx1: Minimum vertex (y_min, x_min) of the axis aligned bounding box.
        yx2: Maximum vertex (y_max, x_max) of the AABB.
        fill: RGB color to draw the corners.
        length: Corner tick length in pixels. Clamped per axis to the box
            width/height, so a large value degrades to a full rectangle.
        width: Line width.
    """
    y1, x1 = map(float, yx1)
    y2, x2 = map(float, yx2)
    if y2 < y1 or x2 < x1:
        raise ValueError(
            f"yx1 must be the min vertex and yx2 the max, but got {yx1=}, {yx2=}"
        )

    draw = PIL.ImageDraw.Draw(image)
    pil_fill = get_pil_ink(fill)

    tick_x = float(min(length, x2 - x1))
    tick_y = float(min(length, y2 - y1))
    for corner_y, corner_x, dir_y, dir_x in [
        (y1, x1, 1, 1),
        (y1, x2, 1, -1),
        (y2, x1, -1, 1),
        (y2, x2, -1, -1),
    ]:
        end_x = corner_x + dir_x * tick_x
        end_y = corner_y + dir_y * tick_y
        draw.line([corner_x, corner_y, end_x, corner_y], fill=pil_fill, width=width)
        draw.line([corner_x, corner_y, corner_x, end_y], fill=pil_fill, width=width)
