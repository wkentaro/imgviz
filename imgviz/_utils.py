from __future__ import annotations

import numpy as np
import PIL.Image
from numpy.typing import NDArray


def pillow_to_numpy(image: PIL.Image.Image) -> NDArray[np.uint8]:
    """Convert Pillow image to numpy array."""
    image_numpy = np.asarray(image)
    if not image_numpy.flags.writeable:
        image_numpy = np.array(image)
    return image_numpy


def numpy_to_pillow(image: NDArray, mode: str | None = None) -> PIL.Image.Image:
    """Convert numpy array to Pillow image."""
    image_pillow = PIL.Image.fromarray(image, mode=mode)
    return image_pillow
