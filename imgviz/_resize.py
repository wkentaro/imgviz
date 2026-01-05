from __future__ import annotations

from typing import Final
from typing import Literal

import numpy as np
import PIL.Image
from numpy.typing import NDArray

from . import _utils

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


def _resize_pillow(
    image: NDArray, height: int, width: int, interpolation: Literal["linear", "nearest"]
) -> NDArray:
    resample: Final
    if interpolation == "linear":
        resample = PIL.Image.BILINEAR
    elif interpolation == "nearest":
        resample = PIL.Image.NEAREST
    else:
        raise ValueError(f"unsupported interpolation: {interpolation}")

    if np.issubdtype(image.dtype, np.integer):
        dst = _utils.numpy_to_pillow(image)
        dst = dst.resize((width, height), resample=resample)
        dst = _utils.pillow_to_numpy(dst)
    else:
        if not np.issubdtype(image.dtype, np.floating):
            raise TypeError(
                f"image.dtype must be integer or floating, got {image.dtype}"
            )
        ndim = image.ndim
        if ndim == 2:
            image = image[:, :, None]

        C = image.shape[2]
        dst = np.zeros((height, width, C), dtype=image.dtype)
        for c in range(C):
            image_c = image[:, :, c]
            image_c = _utils.numpy_to_pillow(image_c)
            dst[:, :, c] = image_c.resize((width, height), resample=resample)

        if ndim == 2:
            dst = dst[:, :, 0]
    return dst


def _resize_opencv(
    image: NDArray, height: int, width: int, interpolation: Literal["linear", "nearest"]
) -> NDArray:
    if cv2 is None:
        raise ImportError("opencv-python is not installed")

    interpolation_int: int
    if interpolation == "linear":
        interpolation_int = cv2.INTER_LINEAR
    elif interpolation == "nearest":
        interpolation_int = cv2.INTER_NEAREST
    else:
        raise ValueError(f"unsupported interpolation: {interpolation}")

    dst = cv2.resize(image, (width, height), interpolation=interpolation_int)
    return dst


def resize(
    image: NDArray,
    height: int | float | None = None,
    width: int | float | None = None,
    interpolation: Literal["linear", "nearest"] = "linear",
    backend: Literal["auto", "pillow", "opencv"] = "auto",
) -> NDArray:
    """Resize image.

    Args:
        image: Input image with shape (H, W) or (H, W, C).
        height: Height of image. If not given, resized based on width keeping
            ratio.
        width: Width of image. If not given, resized based on height keeping
            ratio.
        interpolation: Resizing interpolation ('linear' or 'nearest').
        backend: Resizing backend ('auto', 'pillow', or 'opencv').

    Returns:
        Resized image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image type must be numpy.ndarray")

    if backend == "auto":
        backend = "pillow" if cv2 is None else "opencv"

    image_height, image_width = image.shape[:2]
    if isinstance(width, float):
        scale_width = width
        width = int(round(scale_width * image_width))
    if isinstance(height, float):
        scale_height = height
        height = int(round(scale_height * image_height))
    if height is None and width is None:
        raise ValueError("either height or width must be given")
    if height is None:
        scale_height = width / image_width
        height = int(round(scale_height * image_height))
    if width is None:
        scale_width = height / image_height
        width = int(round(scale_width * image_width))

    if backend == "pillow":
        dst = _resize_pillow(image, height, width, interpolation)
    elif backend == "opencv":
        dst = _resize_opencv(image, height, width, interpolation)
    else:
        raise ValueError(f"unsupported backend: {backend}")

    return dst
