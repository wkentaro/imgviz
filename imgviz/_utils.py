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


def apply_mask(
    image: NDArray,
    transformed: NDArray,
    mask: NDArray[np.bool_] | None,
) -> NDArray:
    """Composite transformed pixels onto image within a mask.

    If mask is None, returns transformed unchanged. Otherwise validates the
    mask and returns a copy of image whose pixels inside the mask are replaced
    by transformed and whose remaining pixels are byte-identical to image.
    """
    if mask is None:
        return transformed
    if mask.dtype != np.bool_:
        raise ValueError(f"mask.dtype must be bool, got {mask.dtype}")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask.shape must be {image.shape[:2]}, got {mask.shape}")

    dst = image.copy()
    dst[mask] = transformed[mask]
    return dst
