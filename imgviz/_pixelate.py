from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import _utils
from ._resize import resize


def pixelate(
    image: NDArray,
    block: int = 8,
    mask: NDArray[np.bool_] | None = None,
) -> NDArray:
    """Pixelate an image into a mosaic.

    Downsamples the image by ``block`` then upscales with nearest-neighbor
    interpolation, producing a mosaic effect.

    Args:
        image: Input image with shape (H, W) or (H, W, C).
        block: Block size in pixels.
        mask: Optional boolean mask (H, W). If given, only pixels inside the
            mask are pixelated and the rest are byte-identical to the input
            (e.g. for redaction). If None, the whole image is pixelated.

    Returns:
        Pixelated image with the same shape and dtype as the input.

    Example:
        >>> import numpy as np
        >>> import imgviz
        >>> image = imgviz.data.arc2017()["rgb"]
        >>> pixelated = imgviz.pixelate(image, block=8)
        >>> mask = np.zeros(image.shape[:2], dtype=bool)
        >>> mask[50:150, 100:250] = True
        >>> redacted = imgviz.pixelate(image, block=8, mask=mask)
    """
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")

    pixelated = _pixelate(image=image, block=block)
    return _utils.apply_mask(image=image, transformed=pixelated, mask=mask)


def _pixelate(image: NDArray, block: int) -> NDArray:
    H, W = image.shape[:2]
    h = max(H // block, 1)
    w = max(W // block, 1)
    small = resize(image, height=h, width=w, interpolation="linear")
    big = resize(small, height=H, width=W, interpolation="nearest")
    return big
