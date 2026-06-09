from collections.abc import Sequence

import numpy as np
import PIL.Image
import PIL.ImageDraw
from numpy.typing import NDArray

from .. import _utils
from ._ink import Ink
from ._ink import get_pil_ink


def pie(
    image: NDArray[np.uint8],
    center: tuple[float, float],
    diameter: float,
    fills: Sequence[Ink],
    outline: Ink | None = None,
    width: int = 0,
) -> NDArray[np.uint8]:
    """Draw a disc divided into equal colored wedges on a numpy array.

    Args:
        image: Input image.
        center: Center point (cy, cx).
        diameter: Diameter of the disc in pixels.
        fills: Fill color for each wedge, drawn in clockwise order starting
            from 12 o'clock. If empty, the image is returned unchanged (any
            outline is ignored). If a single color is given, a full filled
            circle is drawn.
        outline: RGB color for the disc and wedge edges. None for no outline.
        width: Outline width in pixels.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(image)
    pie_(
        image=dst,
        center=center,
        diameter=diameter,
        fills=fills,
        outline=outline,
        width=width,
    )
    return _utils.pillow_to_numpy(dst)


def pie_(
    image: PIL.Image.Image,
    center: tuple[float, float],
    diameter: float,
    fills: Sequence[Ink],
    outline: Ink | None = None,
    width: int = 0,
) -> None:
    """Draw a disc divided into equal colored wedges on a PIL image in-place.

    Args:
        image: PIL image to draw on (modified in-place).
        center: Center point (cy, cx).
        diameter: Diameter of the disc in pixels.
        fills: Fill color for each wedge, drawn in clockwise order starting
            from 12 o'clock. If empty, this is a no-op (any outline is
            ignored). If a single color is given, a full filled circle is
            drawn.
        outline: RGB color for the disc and wedge edges. None for no outline.
        width: Outline width in pixels.
    """
    n = len(fills)
    if n == 0:
        return

    draw = PIL.ImageDraw.Draw(image)

    cy, cx = center
    radius = diameter / 2.0
    bbox = ((cx - radius, cy - radius), (cx + radius, cy + radius))
    pil_outline = get_pil_ink(outline)

    step = 360.0 / n
    for k, fill in enumerate(fills):
        start = -90.0 + k * step
        draw.pieslice(
            bbox,
            start=start,
            end=start + step,
            fill=get_pil_ink(fill),
            outline=pil_outline,
            width=width,
        )
