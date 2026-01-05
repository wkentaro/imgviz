import numpy as np
import PIL.Image
import PIL.ImageDraw
from numpy.typing import NDArray

from .. import _utils
from ._ink import Ink
from ._ink import get_pil_ink


def circle(
    src: NDArray[np.uint8],
    center: tuple[float, float],
    diameter: float,
    fill: Ink | None = None,
    outline: Ink | None = None,
    width: int = 0,
) -> NDArray[np.uint8]:
    """Draw circle on numpy array with Pillow.

    Args:
        src: Input image.
        center: Center point (cy, cx).
        diameter: Diameter of the circle.
        fill: RGB color to fill the mark. None for no fill.
        outline: RGB color to draw the outline.
        width: Line width.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(src)
    circle_(
        image=dst,
        center=center,
        diameter=diameter,
        fill=fill,
        outline=outline,
        width=width,
    )
    return _utils.pillow_to_numpy(dst)


def circle_(
    image: PIL.Image.Image,
    center: tuple[float, float],
    diameter: float,
    fill: Ink | None = None,
    outline: Ink | None = None,
    width: int = 0,
) -> None:
    """Draw circle on PIL image in-place.

    Args:
        image: PIL image to draw on (modified in-place).
        center: Center point (cy, cx).
        diameter: Diameter of the circle.
        fill: RGB color to fill the mark. None for no fill.
        outline: RGB color to draw the outline.
        width: Line width.
    """
    draw = PIL.ImageDraw.Draw(image)

    cy, cx = center

    radius = diameter / 2.0
    x1 = cx - radius
    x2 = x1 + diameter
    y1 = cy - radius
    y2 = y1 + diameter

    draw.ellipse(
        [x1, y1, x2, y2],
        fill=get_pil_ink(fill),
        outline=get_pil_ink(outline),
        width=width,
    )
