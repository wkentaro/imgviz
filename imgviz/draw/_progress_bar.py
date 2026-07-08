import math

import numpy as np
import PIL.Image
from numpy.typing import NDArray

from .. import _utils
from ._ink import Ink
from ._rectangle import rectangle_


def progress_bar(
    image: NDArray[np.uint8],
    yx1: tuple[float, float] | NDArray[np.floating],
    yx2: tuple[float, float] | NDArray[np.floating],
    value: float,
    fill: Ink,
    background: Ink,
    outline: Ink | None = None,
    width: int = 1,
) -> NDArray[np.uint8]:
    """Draw a horizontal progress bar on numpy array with Pillow.

    Args:
        image: Input image.
        yx1: Minimum vertex (y_min, x_min) of the bar's bounding box.
        yx2: Maximum vertex (y_max, x_max) of the bar's bounding box.
        value: Progress in [0, 1]. Out-of-range values are clamped; a
            non-finite value raises ValueError.
        fill: RGB color of the filled portion.
        background: RGB color of the unfilled track.
        outline: RGB color of the bar outline. None for no outline.
        width: Outline width in pixels.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(image)
    progress_bar_(
        image=dst,
        yx1=yx1,
        yx2=yx2,
        value=value,
        fill=fill,
        background=background,
        outline=outline,
        width=width,
    )
    return _utils.pillow_to_numpy(dst)


def progress_bar_(
    image: PIL.Image.Image,
    yx1: tuple[float, float] | NDArray[np.floating],
    yx2: tuple[float, float] | NDArray[np.floating],
    value: float,
    fill: Ink,
    background: Ink,
    outline: Ink | None = None,
    width: int = 1,
) -> None:
    """Draw a horizontal progress bar on PIL image in-place.

    Args:
        image: PIL image to draw on (modified in-place).
        yx1: Minimum vertex (y_min, x_min) of the bar's bounding box.
        yx2: Maximum vertex (y_max, x_max) of the bar's bounding box.
        value: Progress in [0, 1]. Out-of-range values are clamped; a
            non-finite value raises ValueError.
        fill: RGB color of the filled portion.
        background: RGB color of the unfilled track.
        outline: RGB color of the bar outline. None for no outline.
        width: Outline width in pixels.
    """
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, but got: {value}")
    value = max(0.0, min(1.0, value))
    y1, x1 = map(float, yx1)
    y2, x2 = map(float, yx2)

    rectangle_(image, yx1=(y1, x1), yx2=(y2, x2), fill=background)
    if value > 0:
        # Skip at value == 0 to avoid a zero-width rectangle.
        x_fill = x1 + value * (x2 - x1)
        rectangle_(image, yx1=(y1, x1), yx2=(y2, x_fill), fill=fill)
    if outline is not None:
        rectangle_(image, yx1=(y1, x1), yx2=(y2, x2), outline=outline, width=width)
