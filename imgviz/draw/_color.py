import typing
from typing import TypeAlias

import numpy as np
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
