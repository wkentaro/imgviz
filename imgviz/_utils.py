from __future__ import annotations

import numpy as np
import PIL.Image
from numpy.typing import NDArray


def pillow_to_numpy(img: PIL.Image.Image) -> NDArray:
    """Convert Pillow image to numpy array."""
    img_numpy = np.asarray(img)
    if not img_numpy.flags.writeable:
        img_numpy = np.array(img)
    return img_numpy


def numpy_to_pillow(img: NDArray, mode: str | None = None) -> PIL.Image.Image:
    """Convert numpy array to Pillow image."""
    img_pillow = PIL.Image.fromarray(img, mode=mode)
    return img_pillow
