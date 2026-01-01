from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bool2ubyte(img: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """Convert boolean image to uint8."""
    assert img.dtype == bool, "img dtype must be bool"
    return img.astype(np.uint8) * 255


def float2ubyte(img: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Convert float image in [0, 1] to uint8."""
    assert np.issubdtype(img.dtype, float), "img dtype must be float"
    assert img.min() >= 0, "img.min() must be >= 0"
    assert img.max() <= 1, "img.max() must be <= 1"
    return (img * 255).round().astype(np.uint8)
