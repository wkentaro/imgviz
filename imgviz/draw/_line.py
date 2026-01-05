import numpy as np
import PIL.Image
import PIL.ImageDraw
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from .. import _utils
from ._ink import Ink
from ._ink import get_pil_ink


def line(
    src: NDArray[np.uint8],
    yx: ArrayLike,
    fill: Ink,
    width: int = 1,
) -> NDArray[np.uint8]:
    """Draw line on numpy array with Pillow.

    Args:
        src: Input image.
        yx: Array of points (y, x) with shape (N, 2).
        fill: RGB color to draw the line.
        width: Line width.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(src)
    line_(img=dst, yx=yx, fill=fill, width=width)
    return _utils.pillow_to_numpy(dst)


def line_(
    img: PIL.Image.Image,
    yx: ArrayLike,
    fill: Ink,
    width: int = 1,
) -> None:
    """Draw line on PIL image in-place.

    Args:
        img: PIL image to draw on (modified in-place).
        yx: Array of points (y, x) with shape (N, 2).
        fill: RGB color to draw the line.
        width: Line width.
    """
    if not isinstance(img, PIL.Image.Image):
        raise TypeError(f"img must be PIL.Image.Image, but got {type(img).__name__}")
    yx = np.asarray(yx)
    if yx.ndim != 2:
        raise ValueError(f"yx must be 2D array, but got {yx.ndim}D")
    if yx.shape[1] != 2:
        raise ValueError(f"yx.shape[1] must be 2, but got {yx.shape[1]}")

    draw = PIL.ImageDraw.Draw(img)

    xy = yx[:, ::-1]
    xy = xy.flatten().tolist()
    draw.line(xy, fill=get_pil_ink(fill), width=width)
