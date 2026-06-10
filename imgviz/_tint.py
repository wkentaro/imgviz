from __future__ import annotations

from typing import TypeVar
from typing import overload

import cmap as _cmap
import numpy as np
from numpy.typing import NDArray

_FloatT = TypeVar("_FloatT", bound=np.floating)


@overload
def tint(
    image: NDArray[np.uint8], color: _cmap.ColorLike, alpha: float = ...
) -> NDArray[np.uint8]: ...


@overload
def tint(
    image: NDArray[_FloatT], color: _cmap.ColorLike, alpha: float = ...
) -> NDArray[_FloatT]: ...


def tint(image: NDArray, color: _cmap.ColorLike, alpha: float = 0.3) -> NDArray:
    """Wash a whole image toward a solid color at a given opacity.

    A one-liner for flagging thumbnails in tile sheets, e.g. red-tinting a
    rejected comparison. The default ``alpha=0.3`` is a soft 30% wash; raise it
    toward 0.5 for stronger flagging or lower it toward 0.1 for a subtle tint.

    Args:
        image: RGB image with shape (H, W, 3), either uint8 in [0, 255] or
            float in [0, 1].
        color: Wash color, any cmap-compatible value: a name ("red"), hex
            ("#ff0000"), or (r, g, b) tuple of ints in [0, 255], e.g.
            (255, 0, 0), or floats in [0, 1], e.g. (1.0, 0.0, 0.0).
        alpha: Wash opacity in [0, 1]; 0 returns the input, 1 the solid color.

    Returns:
        Tinted image with the same shape and dtype as the input.

    Example:
        >>> import imgviz
        >>> image = imgviz.data.arc2017()["rgb"]
        >>> flagged = imgviz.tint(image, "red", alpha=0.3)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be RGB with shape (H, W, 3), got {image.shape}")
    is_float = np.issubdtype(image.dtype, np.floating)
    if image.dtype != np.uint8 and not is_float:
        raise ValueError(f"image must be uint8 or float, got {image.dtype}")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if alpha == 0:
        return image.copy()

    rgba = _cmap.Color(color).rgba if is_float else _cmap.Color(color).rgba8
    solid = np.array(rgba[:3], dtype=np.float64)
    blended = (1 - alpha) * image + alpha * solid
    if is_float:
        return blended.astype(image.dtype)
    return blended.round().astype(np.uint8)
