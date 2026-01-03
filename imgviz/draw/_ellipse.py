import numpy as np
import PIL.Image
import PIL.ImageDraw
from numpy.typing import NDArray

from .. import _utils
from ._ink import Ink
from ._ink import get_pil_ink


def ellipse(
    src: NDArray[np.uint8],
    aabb1: tuple[float, float],
    aabb2: tuple[float, float],
    fill: Ink | None = None,
    outline: Ink | None = None,
    width: int = 0,
) -> NDArray[np.uint8]:
    dst = _utils.numpy_to_pillow(src)
    ellipse_(img=dst, aabb1=aabb1, aabb2=aabb2, fill=fill, outline=outline, width=width)
    return _utils.pillow_to_numpy(dst)


def ellipse_(
    img: PIL.Image.Image,
    aabb1: tuple[float, float],
    aabb2: tuple[float, float],
    fill: Ink | None = None,
    outline: Ink | None = None,
    width: int = 0,
) -> None:
    draw = PIL.ImageDraw.Draw(img)

    y1, x1 = aabb1
    y2, x2 = aabb2

    draw.ellipse(
        [x1, y1, x2, y2],
        fill=get_pil_ink(fill),
        outline=get_pil_ink(outline),
        width=width,
    )
