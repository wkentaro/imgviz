from __future__ import annotations

from typing import Final
from typing import Literal

import numpy as np
import PIL.Image
from numpy.typing import NDArray

from . import utils

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


def _resize_pillow(
    src: NDArray, height: int, width: int, interpolation: Literal["linear", "nearest"]
) -> NDArray:
    resample: Final
    if interpolation == "linear":
        resample = PIL.Image.BILINEAR
    elif interpolation == "nearest":
        resample = PIL.Image.NEAREST
    else:
        raise ValueError(f"unsupported interpolation: {interpolation}")

    if np.issubdtype(src.dtype, np.integer):
        dst = utils.numpy_to_pillow(src)
        dst = dst.resize((width, height), resample=resample)
        dst = utils.pillow_to_numpy(dst)
    else:
        assert np.issubdtype(src.dtype, np.floating)
        ndim = src.ndim
        if ndim == 2:
            src = src[:, :, None]

        C = src.shape[2]
        dst = np.zeros((height, width, C), dtype=src.dtype)
        for c in range(C):
            src_c = src[:, :, c]
            src_c = utils.numpy_to_pillow(src_c)
            dst[:, :, c] = src_c.resize((width, height), resample=resample)

        if ndim == 2:
            dst = dst[:, :, 0]
    return dst


def _resize_opencv(
    src: NDArray, height: int, width: int, interpolation: Literal["linear", "nearest"]
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

    dst = cv2.resize(src, (width, height), interpolation=interpolation_int)
    return dst


def resize(
    src: NDArray,
    height: int | float | None = None,
    width: int | float | None = None,
    interpolation: Literal["linear", "nearest"] = "linear",
    backend: Literal["auto", "pillow", "opencv"] = "auto",
) -> NDArray:
    """Resize image.

    Parameters
    ----------
    src
        Input image with shape (H, W) or (H, W, C).
    height
        Height of image. If not given, resized based on width keeping ratio.
    width
        Width of image. If not given, resized based on height keeping ratio.
    interpolation
        Resizing interpolation ('linear' or 'nearest').
    backend
        Resizing backend ('auto', 'pillow', or 'opencv').

    Returns
    -------
    dst
        Resized image.

    """
    if not isinstance(src, np.ndarray):
        raise TypeError("src type must be numpy.ndarray")

    if backend == "auto":
        backend = "pillow" if cv2 is None else "opencv"

    src_height, src_width = src.shape[:2]
    if isinstance(width, float):
        scale_width = width
        width = int(round(scale_width * src_width))
    if isinstance(height, float):
        scale_height = height
        height = int(round(scale_height * src_height))
    if height is None:
        assert width is not None
        scale_height = 1.0 * width / src_width
        height = int(round(scale_height * src_height))
    if width is None:
        assert height is not None
        scale_width = 1.0 * height / src_height
        width = int(round(scale_width * src_width))

    if backend == "pillow":
        dst = _resize_pillow(src, height, width, interpolation)
    elif backend == "opencv":
        dst = _resize_opencv(src, height, width, interpolation)
    else:
        raise ValueError(f"unsupported backend: {backend}")

    return dst
