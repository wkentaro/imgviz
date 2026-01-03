from __future__ import annotations

import typing
import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@typing.overload
def normalize(
    src: ...,
    min_value: float | NDArray | None = ...,
    max_value: float | NDArray | None = ...,
    return_minmax: Literal[False] = ...,
) -> NDArray: ...


@typing.overload
def normalize(
    src: ...,
    min_value: float | NDArray | None = ...,
    max_value: float | NDArray | None = ...,
    return_minmax: Literal[True] = ...,
) -> tuple[NDArray, NDArray, NDArray]: ...


def normalize(
    src: NDArray,
    min_value: float | NDArray | None = None,
    max_value: float | NDArray | None = None,
    return_minmax: bool = False,
) -> NDArray | tuple[NDArray, NDArray, NDArray]:
    """Normalize image.

    Args:
        src: Input image with shape (H, W) or (H, W, C).
        min_value: Minimum value.
        max_value: Maximum value.
        return_minmax: Whether to return min_value and max_value.

    Returns:
        Normalized image in [0, 1], or tuple of (dst, min_value, max_value).
    """
    if src.ndim == 2:
        D = 1
    elif src.ndim == 3:
        D = src.shape[2]
    else:
        raise ValueError(f"src ndim must be 2 or 3, but got {src.ndim}")

    if min_value is None:
        min_value = np.nanmin(src, axis=(0, 1))
    min_value = np.atleast_1d(min_value).astype(float)
    if min_value.shape != (D,):
        raise ValueError(f"min_value.shape must be ({D},), but got {min_value.shape}")

    if max_value is None:
        max_value = np.nanmax(src, axis=(0, 1))
    max_value = np.atleast_1d(max_value).astype(float)
    if max_value.shape != (D,):
        raise ValueError(f"max_value.shape must be ({D},), but got {max_value.shape}")

    if np.isinf(min_value).any() or np.isinf(max_value).any():
        warnings.warn("some of min or max values are inf.")

    eps = np.finfo(src.dtype).eps
    issame = max_value == min_value
    min_value[issame] -= eps
    max_value[issame] += eps

    dst = np.zeros(src.shape, dtype=float)

    if src.ndim == 2:
        isnan = np.isnan(src)
    else:
        isnan = np.isnan(src).any(axis=2)
    dst[~isnan] = 1.0 * (src[~isnan] - min_value) / (max_value - min_value)
    dst[isnan] = np.nan

    if return_minmax:
        return dst, min_value, max_value
    else:
        return dst
