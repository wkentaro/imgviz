import numpy as np
import PIL.Image
from numpy.typing import NDArray

from .. import _utils
from ._ink import Ink
from ._ink import require_fill_or_outline
from ._polygon import polygon_


def rotated_rectangle(
    image: NDArray[np.uint8],
    center: tuple[float, float] | NDArray[np.floating],
    size: tuple[float, float] | NDArray[np.floating],
    angle: float,
    fill: Ink | None = None,
    outline: Ink | None = None,
    width: int = 1,
) -> NDArray[np.uint8]:
    """Draw rotated rectangle on numpy array with Pillow.

    Args:
        image: Input image.
        center: Center point (cy, cx).
        size: Rectangle size (height, width) before rotation.
        angle: Rotation angle in degrees. Positive rotates clockwise because the
            image y-axis points down.
        fill: RGB color to fill the mark. None for no fill.
        outline: RGB color to draw the outline.
        width: Outline width.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(image)
    rotated_rectangle_(
        image=dst,
        center=center,
        size=size,
        angle=angle,
        fill=fill,
        outline=outline,
        width=width,
    )
    return _utils.pillow_to_numpy(dst)


def rotated_rectangle_(
    image: PIL.Image.Image,
    center: tuple[float, float] | NDArray[np.floating],
    size: tuple[float, float] | NDArray[np.floating],
    angle: float,
    fill: Ink | None = None,
    outline: Ink | None = None,
    width: int = 1,
) -> None:
    """Draw rotated rectangle on PIL image in-place.

    Args:
        image: PIL image to draw on (modified in-place).
        center: Center point (cy, cx).
        size: Rectangle size (height, width) before rotation.
        angle: Rotation angle in degrees. Positive rotates clockwise because the
            image y-axis points down.
        fill: RGB color to fill the mark. None for no fill.
        outline: RGB color to draw the outline.
        width: Outline width.
    """
    require_fill_or_outline(fill, outline)

    cy, cx = center
    size_h, size_w = size

    corners = np.array(
        [
            [-size_w / 2, -size_h / 2],
            [size_w / 2, -size_h / 2],
            [size_w / 2, size_h / 2],
            [-size_w / 2, size_h / 2],
        ]
    )
    theta = np.deg2rad(angle)
    cos = np.cos(theta)
    sin = np.sin(theta)
    rotation = np.array([[cos, -sin], [sin, cos]])

    xy = corners @ rotation.T + (cx, cy)
    yx = xy[:, ::-1]
    polygon_(image, yx=yx, fill=fill, outline=outline, width=width)
