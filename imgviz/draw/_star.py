import numpy as np
import PIL.Image
from numpy.typing import NDArray

from .. import _utils
from ._ink import Ink
from ._ink import get_draw
from ._ink import get_pil_ink
from ._ink import require_fill_or_outline


def star(
    image: NDArray[np.uint8],
    center: tuple[float, float],
    size: float,
    fill: Ink | None = None,
    outline: Ink | None = None,
    width: int = 1,
) -> NDArray[np.uint8]:  # noqa: GR005 -- preserve positional and keyword API calls
    """Draw star on numpy array with Pillow.

    Args:
        image: Input image.
        center: Center point (cy, cx).
        size: Diameter to create the star.
        fill: RGB color to fill the mark. None for no fill.
        outline: RGB color to draw the outline.
        width: Line width.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(image=image)
    star_(
        image=dst,
        center=center,
        size=size,
        fill=fill,
        outline=outline,
        width=width,
    )
    return _utils.pillow_to_numpy(image=dst)


def star_(
    image: PIL.Image.Image,
    center: tuple[float, float],
    size: float,
    fill: Ink | None = None,
    outline: Ink | None = None,
    width: int = 1,
) -> None:  # noqa: GR005 -- preserve positional and keyword API calls
    """Draw star on PIL image in-place.

    Args:
        image: PIL image to draw on (modified in-place).
        center: Center point (cy, cx).
        size: Diameter to create the star.
        fill: RGB color to fill the mark. None for no fill.
        outline: RGB color to draw the outline.
        width: Line width.
    """
    require_fill_or_outline(fill=fill, outline=outline)

    draw = get_draw(image=image)

    radius = size / 2
    cy, cx = center

    # 5 mountains
    angles_m = np.arange(0, 5) * np.pi * 2 / 5 + (np.pi / 2)
    x_m = cx + radius * np.cos(angles_m)
    y_m = cy - radius * np.sin(angles_m)
    xy_m = np.stack((x_m, y_m), axis=1)

    # 5 valleys
    angles_v = angles_m + np.pi / 5
    length = radius / (np.sin(np.pi / 5) / np.tan(np.pi / 10) + np.cos(np.pi / 10))
    x_v = cx + length * np.cos(angles_v)
    y_v = cy - length * np.sin(angles_v)
    xy_v = np.stack((x_v, y_v), axis=1)

    xy = np.array(
        [
            xy_m[0],
            xy_v[0],
            xy_m[1],
            xy_v[1],
            xy_m[2],
            xy_v[2],
            xy_m[3],
            xy_v[3],
            xy_m[4],
            xy_v[4],
            xy_m[0],
        ]
    )
    xy = xy.flatten().tolist()
    draw.polygon(
        xy, fill=get_pil_ink(ink=fill), outline=get_pil_ink(ink=outline), width=width
    )
