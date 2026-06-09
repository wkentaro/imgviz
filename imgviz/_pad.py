from __future__ import annotations

import numbers
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Color: TypeAlias = int | float | tuple[int | float, ...] | NDArray


def pad(
    image: NDArray,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    color: Color = 0,
) -> NDArray:
    """Pad an image with a constant color on each side.

    Handles ``HW`` and ``HWC`` layouts (including RGBA) and preserves dtype, so
    it replaces ``np.pad(img, ((t, b), (l, r), (0, 0)), constant_values=...)``.

    Args:
        image: Image to pad, with shape (H, W) or (H, W, C).
        top: Pixels to add above the image.
        bottom: Pixels to add below the image.
        left: Pixels to add to the left of the image.
        right: Pixels to add to the right of the image.
        color: Fill color for the added border. A scalar fills every channel; a
            tuple or array fills per channel and must match the channel count.

    Returns:
        A new padded image with the same dtype as the input.

    Example:
        >>> import imgviz
        >>> image = imgviz.data.arc2017()["rgb"]
        >>> framed = imgviz.pad(image, top=8, bottom=8, left=8, right=8)
        >>> banner = imgviz.pad(image, top=40, color=(0, 0, 0))
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"image.ndim must be 2 or 3, but got {image.ndim}")

    for name, value in (
        ("top", top),
        ("bottom", bottom),
        ("left", left),
        ("right", right),
    ):
        if not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} must be int, but got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be >= 0, but got {value}")

    components = np.asarray(color)
    if components.ndim > 0:
        if image.ndim == 2:
            raise ValueError(
                f"color has {components.size} components but a single-channel "
                "(H, W) image takes a scalar color"
            )
        if components.size != image.shape[2]:
            raise ValueError(
                f"color has {components.size} components but image has "
                f"{image.shape[2]} channels"
            )

    h, w = image.shape[:2]
    shape = (h + top + bottom, w + left + right) + image.shape[2:]
    dst = np.empty(shape, dtype=image.dtype)
    dst[...] = color
    dst[top : top + h, left : left + w] = image
    return dst
