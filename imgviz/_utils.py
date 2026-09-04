from __future__ import annotations

from typing import Literal

import numpy as np
import PIL.Image
from numpy.typing import NDArray


def compute_corner_origin(
    container_size: tuple[int, int],
    block_size: tuple[int, int],
    loc: Literal["lt", "rt", "lb", "rb"],
    margin: int,
) -> tuple[int, int]:
    """Top-left origin (y0, x0) for placing a block in a container corner.

    Places a block of ``block_size`` inside a ``container_size`` region, offset
    by ``margin`` from the corner selected by ``loc`` ("lt", "rt", "lb", "rb").
    """
    if loc not in ("lt", "rt", "lb", "rb"):
        raise ValueError(f"unsupported loc: {loc}")
    height, width = container_size
    block_height, block_width = block_size
    y0 = margin if loc in ("lt", "rt") else height - margin - block_height
    x0 = margin if loc in ("lt", "lb") else width - margin - block_width
    return y0, x0


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
