from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bool2ubyte(img: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """Convert boolean image to uint8."""
    if img.dtype != bool:
        raise ValueError(f"img dtype must be bool, but got {img.dtype}")
    return img.astype(np.uint8) * 255


def float2ubyte(img: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Convert float image in [0, 1] to uint8."""
    if not np.issubdtype(img.dtype, np.floating):
        raise ValueError(f"img dtype must be float, but got {img.dtype}")
    if img.min() < 0:
        raise ValueError(f"img.min() must be >= 0, but got {img.min()}")
    if img.max() > 1:
        raise ValueError(f"img.max() must be <= 1, but got {img.max()}")
    return (img * 255).round().astype(np.uint8)
