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

    eps = np.finfo(image.dtype).eps
    issame = max_value == min_value
    min_value[issame] -= eps
    max_value[issame] += eps

    dst: NDArray[np.float32] = np.zeros(image.shape, dtype=np.float32)

    if image.ndim == 2:
        isnan = np.isnan(image)
    else:
        isnan = np.isnan(image).any(axis=2)
    dst[~isnan] = 1.0 * (image[~isnan] - min_value) / (max_value - min_value)
    dst[isnan] = np.nan

    if return_minmax:
        return dst, min_value, max_value
    else:
        return dst
