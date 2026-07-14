import numpy as np
import PIL.Image
import PIL.ImageDraw
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from .. import _utils
from ._ink import Ink
from ._ink import get_pil_ink
from ._ink import require_pil_image


def arrow(
    image: NDArray[np.uint8],
    yx1: ArrayLike,
    yx2: ArrayLike,
    fill: Ink,
    width: int = 1,
    head_length_ratio: float = 0.1,
    head_angle: float = 30.0,
) -> NDArray[np.uint8]:
    """Draw arrow on numpy array with Pillow.

    Args:
        image: Input image.
        yx1: Tail (y, x) where the arrow starts.
        yx2: Tip (y, x) where the arrowhead is drawn.
        fill: RGB color to draw the arrow.
        width: Line width.
        head_length_ratio: Arrowhead length as a fraction of the shaft length.
        head_angle: Half-angle of the arrowhead in degrees.

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(image)
    arrow_(
        image=dst,
        yx1=yx1,
        yx2=yx2,
        fill=fill,
        width=width,
        head_length_ratio=head_length_ratio,
        head_angle=head_angle,
    )
    return _utils.pillow_to_numpy(dst)


def arrow_(
    image: PIL.Image.Image,
    yx1: ArrayLike,
    yx2: ArrayLike,
    fill: Ink,
    width: int = 1,
    head_length_ratio: float = 0.1,
    head_angle: float = 30.0,
) -> None:
    """Draw arrow on PIL image in-place.

    Args:
        image: PIL image to draw on (modified in-place).
        yx1: Tail (y, x) where the arrow starts.
        yx2: Tip (y, x) where the arrowhead is drawn.
        fill: RGB color to draw the arrow.
        width: Line width.
        head_length_ratio: Arrowhead length as a fraction of the shaft length.
        head_angle: Half-angle of the arrowhead in degrees.
    """
    require_pil_image(image=image)
    tail = np.asarray(yx1, dtype=float)
    tip = np.asarray(yx2, dtype=float)
    if tail.shape != (2,):
        raise ValueError(f"yx1 must have shape (2,), but got {tail.shape}")
    if tip.shape != (2,):
        raise ValueError(f"yx2 must have shape (2,), but got {tip.shape}")

    draw = PIL.ImageDraw.Draw(image)
    pil_fill = get_pil_ink(fill)

    y1, x1 = float(tail[0]), float(tail[1])
    y2, x2 = float(tip[0]), float(tip[1])
    draw.line([x1, y1, x2, y2], fill=pil_fill, width=width)

    shaft = tip - tail
    length = float(np.linalg.norm(shaft))
    if length == 0:
        return
    uy, ux = shaft / length
    head_length = length * head_length_ratio
    # rotating the shaft direction by +head_angle and -head_angle yields a barb
    # pair symmetric about the shaft, so the (y, x) rotation's handedness is moot
    for sign in (1, -1):
        a = np.radians(sign * head_angle)
        cos_a, sin_a = np.cos(a), np.sin(a)
        by = y2 - head_length * float(uy * cos_a - ux * sin_a)
        bx = x2 - head_length * float(uy * sin_a + ux * cos_a)
        draw.line([x2, y2, bx, by], fill=pil_fill, width=width)
