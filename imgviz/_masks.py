from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._color import asrgb
from .fill import Fill
from .fill import Solid


def mask2rgb(
    mask: NDArray[np.bool_],
    image: NDArray[np.uint8] | None = None,
    fill: Fill | tuple[int, int, int] | NDArray[np.uint8] = (0, 255, 0),
    alpha: float = 0.5,
    cval: tuple[int, int, int] = (0, 0, 0),
) -> NDArray[np.uint8]:
    """Fill mask region with color or pattern.

    Parameters
    ----------
    mask
        Boolean mask (H, W).
    image
        Background image to blend with. If None, returns solid color.
    fill
        Fill specification (e.g., Solid, Stripe). If a type of Fill is not given,
        the given value (e.g., an RGB tuple or NDArray[np.uint8]) is interpreted
        as a color specification for Solid fill.
    alpha
        Opacity of fill (0.0 to 1.0). Only used when image is provided.
    cval
        RGB color for background when image is None. Defaults to black.

    Returns
    -------
    result
        Image with filled mask (H, W, 3).

    """
    if mask.ndim != 2:
        raise ValueError(f"mask.ndim must be 2, got {mask.ndim}")
    if mask.dtype != np.bool_:
        raise ValueError(f"mask.dtype must be bool, got {mask.dtype}")
    if not isinstance(fill, Fill):
        fill = Solid(color=fill)

    if image is None:
        h, w = mask.shape
        image = np.full((h, w, 3), fill_value=cval, dtype=np.uint8)
        alpha = 1.0  # ignore alpha when no image is given
    else:
        image = asrgb(image, copy=True)

    return fill(mask=mask, image=image, alpha=alpha, copy=False)
