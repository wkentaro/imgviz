from __future__ import annotations

import typing
import warnings
from typing import Any
from typing import Final
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._letterbox import letterbox

_DEPRECATION_MESSAGE: Final[str] = (
    "`centerize` is deprecated and will be removed in the next major release; "
    "use `imgviz.letterbox` instead."
)


@typing.overload
def centerize(
    image: NDArray,
    height: int,
    width: int,
    cval: object = ...,
    return_mask: Literal[False] = ...,
    interpolation: Literal["linear", "nearest"] = ...,
    loc: Literal["center", "lt", "rt", "lb", "rb"] = ...,
) -> NDArray: ...


@typing.overload
def centerize(
    image: NDArray,
    height: int,
    width: int,
    cval: object = ...,
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
    """Deprecated. Use :func:`imgviz.letterbox` instead."""
    warnings.warn(
        _DEPRECATION_MESSAGE,
        DeprecationWarning,
        stacklevel=2,
    )
    return letterbox(
        image,
        height=height,
        width=width,
        color=cval,
        return_mask=return_mask,
        interpolation=interpolation,
        loc=loc,
    )
