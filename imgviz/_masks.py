from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import _color


def mask2rgb(
    mask: NDArray[np.bool_],
    image: NDArray[np.uint8] | None = None,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (0, 255, 0),
    cval: tuple[int, int, int] = (0, 0, 0),
) -> NDArray[np.uint8]:
    """Fill mask region with color.

    Parameters
    ----------
    mask
        Boolean mask (H, W).
    image
        Background image to blend with. If None, returns solid color.
    alpha
        Opacity of fill. Only used when image is provided.
    color
        RGB color to fill. Default green (0, 255, 0).
    cval
        RGB color for background when image is None. Default black (0, 0, 0).

    Returns
    -------
    result
        Image with filled mask (H, W, 3).

    """
    if mask.ndim != 2:
        raise ValueError(f"mask.ndim must be 2, got {mask.ndim}")
    if mask.dtype != np.bool_:
        raise ValueError(f"mask.dtype must be bool, got {mask.dtype}")

    result: NDArray[np.uint8]
    if image is None:
        result = np.full(mask.shape + (3,), fill_value=cval, dtype=np.uint8)
        result[mask] = color
    else:
        result = _color.asrgb(img=image, copy=True)
        blended: NDArray[np.float32] = (1 - alpha) * result[mask].astype(
            np.float32
        ) + alpha * np.array(color, dtype=np.float32)
        result[mask] = np.clip(blended.round(), 0, 255).astype(np.uint8)

    return result
