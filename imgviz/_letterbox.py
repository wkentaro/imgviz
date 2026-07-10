from __future__ import annotations

import typing
from typing import Any
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._pad import pad
from ._resize import resize


@typing.overload
def letterbox(
    image: NDArray,
    height: int,
    width: int,
    color: object = ...,
    return_mask: Literal[False] = ...,
    interpolation: Literal["linear", "nearest"] = ...,
    loc: Literal["center", "lt", "rt", "lb", "rb"] = ...,
) -> NDArray: ...


@typing.overload
def letterbox(
    image: NDArray,
    height: int,
    width: int,
    color: object = ...,
    return_mask: Literal[True] = ...,
    interpolation: Literal["linear", "nearest"] = ...,
    loc: Literal["center", "lt", "rt", "lb", "rb"] = ...,
) -> tuple[NDArray, NDArray[np.bool_]]: ...


def letterbox(
    image: NDArray,
    height: int,
    width: int,
    color: Any = None,
    return_mask: bool = False,
    interpolation: Literal["linear", "nearest"] = "linear",
    loc: Literal["center", "lt", "rt", "lb", "rb"] = "center",
) -> NDArray | tuple[NDArray, NDArray[np.bool_]]:
    """Resize image preserving aspect ratio and pad to target size.

    Args:
        image: Image to letterbox.
        height: Target height.
        width: Target width.
        color: Color to fill the padding area.
        return_mask: Whether to return mask for the resized image region.
        interpolation: Interpolation method.
        loc: Where to place the resized image inside the canvas.

    Returns:
        A new letterboxed image (never a view of ``image``), or tuple of
        (image, mask) if return_mask is True.
    """
    if image.ndim == 3:
        shape: tuple[int, ...] = (height, width, image.shape[2])
    else:
        shape = (height, width)

    if image.shape[:2] == shape[:2]:
        if return_mask:
            return image.copy(), np.ones(shape[:2], dtype=bool)
        else:
            return image.copy()

    image_h, image_w = image.shape[:2]
    scale = min(1.0 * height / image_h, 1.0 * width / image_w)
    image = resize(
        image,
        height=max(1, int(round(image_h * scale))),
        width=max(1, int(round(image_w * scale))),
        interpolation=interpolation,
    )

    ph, pw = 0, 0
    h, w = image.shape[:2]
    if loc == "center":
        if h < height:
            ph = (height - h) // 2
        if w < width:
            pw = (width - w) // 2
    elif loc == "lt":
        ph = 0
        pw = 0
    elif loc == "rt":
        ph = 0
        if w < width:
            pw = width - w
    elif loc == "lb":
        if h < height:
            ph = height - h
        pw = 0
    elif loc == "rb":
        if h < height:
            ph = height - h
        if w < width:
            pw = width - w
    else:
        raise ValueError(f"unsupported loc: {loc}")

    dst = pad(
        image,
        top=ph,
        bottom=height - h - ph,
        left=pw,
        right=width - w - pw,
        color=0 if color is None else color,
    )

    if return_mask:
        mask = np.zeros(shape[:2], dtype=bool)
        mask[ph : ph + h, pw : pw + w] = True
        return dst, mask
    else:
        return dst
