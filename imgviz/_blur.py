from __future__ import annotations

import numpy as np
import PIL.Image
import PIL.ImageFilter
from numpy.typing import NDArray

from . import _utils


def blur(
    image: NDArray[np.uint8],
    sigma: float = 8.0,
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.uint8]:
    """Apply Gaussian blur to an image.

    Args:
        image: Input uint8 image with shape (H, W) or (H, W, C).
        sigma: Gaussian blur radius in pixels.
        mask: Optional boolean mask (H, W). If given, only pixels inside the
            mask are blurred and the rest are byte-identical to the input
            (e.g. for redaction). If None, the whole image is blurred.

    Returns:
        Blurred image with the same shape and dtype as the input.

    Example:
        >>> import numpy as np
        >>> import imgviz
        >>> image = imgviz.data.arc2017()["rgb"]
        >>> blurred = imgviz.blur(image, sigma=10)
        >>> mask = np.zeros(image.shape[:2], dtype=bool)
        >>> mask[50:150, 100:250] = True
        >>> redacted = imgviz.blur(image, sigma=10, mask=mask)
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if image.dtype != np.uint8:
        raise ValueError(f"image.dtype must be uint8, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"image.ndim must be 2 or 3, got {image.ndim}")

    blurred = _gaussian_blur(image=image, sigma=sigma)
    return _utils.apply_mask(image=image, transformed=blurred, mask=mask)


def _gaussian_blur(image: NDArray, sigma: float) -> NDArray:
    # Pillow blurs each band independently, so blurring a whole (H, W),
    # (H, W, 2), (H, W, 3), or (H, W, 4) image in one pass is bit-identical to
    # blurring band by band. Other channel counts have no Pillow mode, so they
    # fall back to the per-band loop.
    if image.ndim == 2 or image.shape[2] in (2, 3, 4):
        return _gaussian_blur_pil(arr=image, sigma=sigma)

    dst = np.empty_like(image)
    for c in range(image.shape[2]):
        dst[..., c] = _gaussian_blur_pil(arr=image[..., c], sigma=sigma)
    return dst


def _gaussian_blur_pil(arr: NDArray, sigma: float) -> NDArray:
    pil = PIL.Image.fromarray(arr)
    blurred = pil.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(blurred)
