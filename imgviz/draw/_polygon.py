import numpy as np
import PIL.Image
import PIL.ImageDraw
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from .. import _utils
from ._ink import Ink
from ._ink import get_pil_ink
from ._ink import require_fill_or_outline


def polygon(
    image: NDArray[np.uint8],
    yx: ArrayLike,
    fill: Ink | None = None,
    outline: Ink | None = None,
    width: int = 1,
) -> NDArray[np.uint8]:
    """Draw polygon on numpy array with Pillow.

    Args:
        image: Input image.
        yx: Array of vertices (y, x) with shape (N, 2). The polygon is closed
            automatically; the first vertex does not need to be repeated.
        fill: RGB color to fill the polygon. None for no fill.
        outline: RGB color to draw the outline. None for no outline.
        width: Outline width.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(image)
    polygon_(image=dst, yx=yx, fill=fill, outline=outline, width=width)
    return _utils.pillow_to_numpy(dst)


def polygon_(
    image: PIL.Image.Image,
    yx: ArrayLike,
    fill: Ink | None = None,
    outline: Ink | None = None,
    width: int = 1,
) -> None:
    """Draw polygon on PIL image in-place.

    Args:
        image: PIL image to draw on (modified in-place).
        yx: Array of vertices (y, x) with shape (N, 2). The polygon is closed
            automatically; the first vertex does not need to be repeated.
        fill: RGB color to fill the polygon. None for no fill.
        outline: RGB color to draw the outline. None for no outline.
        width: Outline width.
    """
    if not isinstance(image, PIL.Image.Image):
        raise TypeError(
            f"image must be PIL.Image.Image, but got {type(image).__name__}"
        )
    require_fill_or_outline(fill, outline)
    yx = np.asarray(yx)
    if yx.ndim != 2:
        raise ValueError(f"yx must be 2D array, but got {yx.ndim}D")
    if yx.shape[1] != 2:
        raise ValueError(f"yx.shape[1] must be 2, but got {yx.shape[1]}")
    if yx.shape[0] < 3:
        raise ValueError(f"yx must have at least 3 vertices, but got {yx.shape[0]}")

    xy = yx[:, ::-1].flatten().tolist()
    draw = PIL.ImageDraw.Draw(image)
    draw.polygon(
        xy=xy,
        fill=get_pil_ink(fill),
        outline=get_pil_ink(outline),
        width=width,
    )
