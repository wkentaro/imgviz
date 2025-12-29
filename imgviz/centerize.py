from __future__ import annotations

import typing
from typing import Any
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .resize import resize


@typing.overload
def centerize(
    src: ...,
    shape: tuple[int, ...],
    cval: Any = ...,
    return_mask: Literal[False] = ...,
    interpolation: Literal["linear", "nearest"] = ...,
    loc: Literal["center", "lt", "rb"] = ...,
) -> NDArray: ...


@typing.overload
def centerize(
    src: ...,
    shape: tuple[int, ...],
    cval: Any = ...,
    return_mask: Literal[True] = ...,
    interpolation: Literal["linear", "nearest"] = ...,
    loc: Literal["center", "lt", "rb"] = ...,
) -> tuple[NDArray, NDArray[np.bool_]]: ...


def centerize(
    src: NDArray,
    shape: tuple[int, ...],
    cval: Any = None,
    return_mask: bool = False,
    interpolation: Literal["linear", "nearest"] = "linear",
    loc: Literal["center", "lt", "rb"] = "center",
) -> NDArray | tuple[NDArray, NDArray[np.bool_]]:
    """Centerize image for specified image size.

    Parameters
    ----------
    src
        Image to centerize.
    shape
        Image shape (height, width) or (height, width, channel).
    cval
        Color to be filled in the blank.
    return_mask
        Whether to return mask for centerized image.
    interpolation
        Interpolation method.
    loc
        Location of image.

    Returns
    -------
    dst
        Centerized image, or tuple of (image, mask) if return_mask is True.

    """
    if src.shape[:2] == shape[:2]:
        if return_mask:
            return src, np.ones(shape[:2], dtype=bool)
        else:
            return src

    if len(shape) != src.ndim:
        shape = list(shape) + [src.shape[2]]

    dst = np.zeros(shape, dtype=src.dtype)
    if cval:
        dst[:, :] = cval

    src_h, src_w = src.shape[:2]
    scale_h, scale_w = 1.0 * shape[0] / src_h, 1.0 * shape[1] / src_w
    scale = min(scale_h, scale_w)
    dst_h, dst_w = int(round(src_h * scale)), int(round(src_w * scale))
    src = resize(src, height=dst_h, width=dst_w, interpolation=interpolation)

    ph, pw = 0, 0
    h, w = src.shape[:2]
    dst_h, dst_w = shape[:2]
    if loc == "center":
        if h < dst_h:
            ph = (dst_h - h) // 2
        if w < dst_w:
            pw = (dst_w - w) // 2
    elif loc == "lt":
        ph = 0
        pw = 0
    elif loc == "rb":
        if h < dst_h:
            ph = dst_h - h
        if w < dst_w:
            pw = dst_w - w
    else:
        raise ValueError(f"Unsupported loc: {loc}")
    dst[ph : ph + h, pw : pw + w] = src

    if return_mask:
        mask = np.zeros(shape[:2], dtype=bool)
        mask[ph : ph + h, pw : pw + w] = True
        return dst, mask
    else:
        return dst
