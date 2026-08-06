from __future__ import annotations

import typing
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@typing.overload
def normalize(
    image: NDArray,
    min_value: float | Sequence[float] | NDArray[np.floating] | None = ...,
    max_value: float | Sequence[float] | NDArray[np.floating] | None = ...,
    return_minmax: Literal[False] = ...,
) -> NDArray[np.float32]: ...


@typing.overload
def normalize(
    image: NDArray,
    min_value: float | Sequence[float] | NDArray[np.floating] | None = ...,
    max_value: float | Sequence[float] | NDArray[np.floating] | None = ...,
    return_minmax: Literal[True] = ...,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]: ...


def normalize(
    image: NDArray,
    min_value: float | Sequence[float] | NDArray[np.floating] | None = None,
    max_value: float | Sequence[float] | NDArray[np.floating] | None = None,
    return_minmax: bool = False,
) -> (
    NDArray[np.float32]
    | tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
):
    """Normalize image.

    Args:
        image: Input image with shape (H, W) or (H, W, C).
        min_value: Minimum value.
        max_value: Maximum value.
        return_minmax: Whether to return min_value and max_value.

    Returns:
        Normalized image in [0, 1], or tuple of (dst, min_value, max_value).
    """
    if image.ndim == 2:
        D = 1
    elif image.ndim == 3:
        D = image.shape[2]
    else:
        raise ValueError(f"image ndim must be 2 or 3, but got {image.ndim}")

    if min_value is None:
        min_value = np.nanmin(image, axis=(0, 1))
    min_value = np.atleast_1d(min_value).astype(np.float32)
    if min_value.shape != (D,):
        raise ValueError(f"min_value.shape must be ({D},), but got {min_value.shape}")

    if max_value is None:
        max_value = np.nanmax(image, axis=(0, 1))
    max_value = np.atleast_1d(max_value).astype(np.float32)
    if max_value.shape != (D,):
        raise ValueError(f"max_value.shape must be ({D},), but got {max_value.shape}")

    if np.isinf(min_value).any() or np.isinf(max_value).any():
        warnings.warn("some of min or max values are inf.")

    # Spread proportional to magnitude so the subtraction survives float32
    # rounding even when min_value is large.
    eps = np.finfo(min_value.dtype).eps
    issame = max_value == min_value
    spread = eps * np.maximum(np.abs(min_value[issame]), 1.0)
    min_value[issame] -= spread
    max_value[issame] += spread

    dst: NDArray[np.float32] = ((image - min_value) / (max_value - min_value)).astype(
        np.float32, copy=False
    )
    # For (H, W) input NaN already propagates through the arithmetic; only the
    # multichannel case needs to collapse a per-channel NaN across the pixel.
    if image.ndim == 3:
        isnan = np.isnan(image).any(axis=2)
        if isnan.any():
            dst[isnan] = np.nan

    if return_minmax:
        return dst, min_value, max_value
    else:
        return dst
