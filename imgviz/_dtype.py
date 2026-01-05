from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bool2ubyte(image: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """Convert boolean image to uint8."""
    if image.dtype != bool:
        raise ValueError(f"image dtype must be bool, but got {image.dtype}")
    return image.astype(np.uint8) * 255


def float2ubyte(image: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Convert float image in [0, 1] to uint8."""
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"image dtype must be float, but got {image.dtype}")
    if image.min() < 0:
        raise ValueError(f"image.min() must be >= 0, but got {image.min()}")
    if image.max() > 1:
        raise ValueError(f"image.max() must be <= 1, but got {image.max()}")
    return (image * 255).round().astype(np.uint8)
