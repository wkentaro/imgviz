import typing
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Color: TypeAlias = (
    int | tuple[int, int, int] | tuple[int, int, int, int] | NDArray[np.uint8]
)


def get_pil_color(
    color: Color | None,
) -> int | tuple[int, int, int] | tuple[int, int, int, int] | None:
    if isinstance(color, np.ndarray):
        if color.ndim != 1 or color.size not in (3, 4):
            raise ValueError(
                f"color ndarray must be 1D with size 3 or 4, but got {color.shape=}"
            )
        color = typing.cast(
            tuple[int, int, int] | tuple[int, int, int, int],
            tuple(int(c) for c in color.tolist()),
        )
    return color
