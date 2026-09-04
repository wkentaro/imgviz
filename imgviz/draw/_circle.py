import numpy as np
import PIL.Image
from numpy.typing import NDArray

from .. import _utils
from ._ellipse import ellipse_
from ._ink import Ink


def circle(
    image: NDArray[np.uint8],
    center: tuple[float, float],
    diameter: float,
    fill: Ink | None = None,
    outline: Ink | None = None,
    width: int = 0,
) -> NDArray[np.uint8]:
    """Draw circle on numpy array with Pillow.

    Args:
        image: Input image.
        center: Center point (cy, cx).
        diameter: Diameter of the circle.
        fill: RGB color to fill the mark. None for no fill.
        outline: RGB color to draw the outline.
        width: Line width.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(image)
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
    cy, cx = center

    radius = diameter / 2.0
    x1 = cx - radius
    y1 = cy - radius

    ellipse_(
        image=image,
        yx1=(y1, x1),
        yx2=(y1 + diameter, x1 + diameter),
        fill=fill,
        outline=outline,
        width=width,
    )
