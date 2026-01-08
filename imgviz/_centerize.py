from __future__ import annotations

import typing
from typing import Any
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._resize import resize


@typing.overload
def centerize(
    image: ...,
    height: int,
    width: int,
    cval: Any = ...,
    return_mask: Literal[False] = ...,
    interpolation: Literal["linear", "nearest"] = ...,
    loc: Literal["center", "lt", "rt", "lb", "rb"] = ...,
) -> NDArray: ...


@typing.overload
def centerize(
    image: ...,
    height: int,
    width: int,
    cval: Any = ...,
    return_mask: Literal[True] = ...,
    interpolation: Literal["linear", "nearest"] = ...,
    loc: Literal["center", "lt", "rt", "lb", "rb"] = ...,
) -> tuple[NDArray, NDArray[np.bool_]]: ...


def centerize(
    image: NDArray,
    height: int,
    width: int,
    cval: Any = None,
    return_mask: bool = False,
    interpolation: Literal["linear", "nearest"] = "linear",
    loc: Literal["center", "lt", "rt", "lb", "rb"] = "center",
) -> NDArray | tuple[NDArray, NDArray[np.bool_]]:
    """Centerize image for specified image size.

    Args:
        image: Image to centerize.
        height: Target height.
        width: Target width.
        cval: Color to be filled in the blank.
        return_mask: Whether to return mask for centerized image.
        interpolation: Interpolation method.
        loc: Location of image.

    Returns:
        Centerized image, or tuple of (image, mask) if return_mask is True.
    """
    if image.ndim == 3:
        shape: tuple[int, ...] = (height, width, image.shape[2])
    else:
        shape = (height, width)

    if image.shape[:2] == shape[:2]:
        if return_mask:
            return image, np.ones(shape[:2], dtype=bool)
        else:
            return image

    dst = np.zeros(shape, dtype=image.dtype)
    if cval is not None:
        dst[:, :] = cval

    image_h, image_w = image.shape[:2]
    scale_h, scale_w = 1.0 * shape[0] / image_h, 1.0 * shape[1] / image_w
    scale = min(scale_h, scale_w)
    dst_h, dst_w = int(round(image_h * scale)), int(round(image_w * scale))
    image = resize(image, height=dst_h, width=dst_w, interpolation=interpolation)

    ph, pw = 0, 0
    h, w = image.shape[:2]
    dst_h, dst_w = shape[:2]
    if loc == "center":
        if h < dst_h:
            ph = (dst_h - h) // 2
        if w < dst_w:
            pw = (dst_w - w) // 2
    elif loc == "lt":
        ph = 0
        pw = 0
    elif loc == "rt":
        ph = 0
        if w < dst_w:
            pw = dst_w - w
    elif loc == "lb":
        if h < dst_h:
            ph = dst_h - h
        pw = 0
    elif loc == "rb":
        if h < dst_h:
            ph = dst_h - h
        if w < dst_w:
            pw = dst_w - w
    else:
        raise ValueError(f"unsupported loc: {loc}")
    dst[ph : ph + h, pw : pw + w] = image

    if return_mask:
        mask = np.zeros(shape[:2], dtype=bool)
        mask[ph : ph + h, pw : pw + w] = True
        return dst, mask
    else:
        return dst
