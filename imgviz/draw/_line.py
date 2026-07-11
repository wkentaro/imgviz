import numpy as np
import PIL.Image
import PIL.ImageDraw
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from .. import _utils
from ._ink import Ink
from ._ink import as_yx_points
from ._ink import get_pil_ink
from ._ink import require_pil_image


def line(
    image: NDArray[np.uint8],
    yx: ArrayLike,
    fill: Ink,
    width: int = 1,
) -> NDArray[np.uint8]:
    """Draw line on numpy array with Pillow.

    Args:
        image: Input image.
        yx: Array of points (y, x) with shape (N, 2).
        fill: RGB color to draw the line.
        width: Line width.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(image)
    line_(image=dst, yx=yx, fill=fill, width=width)
    return _utils.pillow_to_numpy(dst)


def line_(
    image: PIL.Image.Image,
    yx: ArrayLike,
    fill: Ink,
    width: int = 1,
) -> None:
    """Draw line on PIL image in-place.

    Args:
        image: PIL image to draw on (modified in-place).
        yx: Array of points (y, x) with shape (N, 2).
        fill: RGB color to draw the line.
        width: Line width.
    """
    require_pil_image(image=image)
    yx = as_yx_points(yx)

    draw = PIL.ImageDraw.Draw(image)

    xy = yx[:, ::-1]
    xy = xy.flatten().tolist()
    draw.line(xy, fill=get_pil_ink(fill), width=width)
