from __future__ import annotations

import numpy as np
import PIL.Image
from numpy.typing import NDArray


def pillow_to_numpy(*, image: PIL.Image.Image) -> NDArray[np.uint8]:
    image_numpy = np.asarray(image)
    if not image_numpy.flags.writeable:
        image_numpy = np.array(image)
    return image_numpy


def numpy_to_pillow(*, image: NDArray, mode: str | None = None) -> PIL.Image.Image:
    image_pillow = PIL.Image.fromarray(image, mode=mode)
    return image_pillow


def apply_mask(
    *,
    image: NDArray,
    transformed: NDArray,
    mask: NDArray[np.bool_] | None,
) -> NDArray:
    if mask is None:
        return transformed
    if mask.dtype != np.bool_:
        raise ValueError(f"mask.dtype must be bool, got {mask.dtype}")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask.shape must be {image.shape[:2]}, got {mask.shape}")

    dst = image.copy()
    dst[mask] = transformed[mask]
    return dst
