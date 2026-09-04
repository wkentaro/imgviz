import typing
from typing import TypeAlias

import numpy as np
import PIL.Image
from numpy.typing import ArrayLike
from numpy.typing import NDArray

Ink: TypeAlias = (
    int | tuple[int, int, int] | tuple[int, int, int, int] | NDArray[np.uint8]
)


def get_pil_ink(
    ink: Ink | None,
) -> int | tuple[int, int, int] | tuple[int, int, int, int] | None:
    if isinstance(ink, np.ndarray):
        if ink.ndim != 1 or ink.size not in (3, 4):
            raise ValueError(
                f"color ndarray must be 1D with size 3 or 4, but got {ink.shape=}"
            )
        ink = typing.cast(
            tuple[int, int, int] | tuple[int, int, int, int],
            tuple(int(c) for c in ink.tolist()),
        )
    return ink


def as_yx_points(yx: ArrayLike) -> NDArray:
    yx = np.asarray(yx)
    if yx.ndim != 2:
        raise ValueError(f"yx must be 2D array, but got {yx.ndim}D")
    if yx.shape[1] != 2:
        raise ValueError(f"yx.shape[1] must be 2, but got {yx.shape[1]}")
    return yx


def require_fill_or_outline(fill: Ink | None, outline: Ink | None) -> None:
    if fill is None and outline is None:
        raise ValueError("at least one of `fill` or `outline` must be set")


def require_pil_image(*, image: PIL.Image.Image) -> None:
    if not isinstance(image, PIL.Image.Image):
        raise TypeError(
            f"image must be PIL.Image.Image, but got {type(image).__name__}"
        )
