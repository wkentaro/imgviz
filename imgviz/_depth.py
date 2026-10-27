from __future__ import annotations

import typing
import warnings
from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ._colorize import Colorize
from ._colorize import colorize

_DEPRECATION_MESSAGE: Final[str] = (
    "`{name}` is deprecated and will be removed in the next major release; "
    "use `imgviz.{replacement}` instead."
)


class Depth2Rgb:
    """Deprecated. Use :class:`imgviz.Colorize` instead."""

    def __init__(
        self,
        min_value: float | NDArray | None = None,
        max_value: float | NDArray | None = None,
        colormap: str | Callable[[NDArray], NDArray] = "jet",
    ) -> None:  # noqa: GR005 -- preserve positional and keyword API calls
        warnings.warn(
            _DEPRECATION_MESSAGE.format(name="Depth2Rgb", replacement="Colorize"),
            DeprecationWarning,
            stacklevel=2,
        )
        self._colorize = Colorize(vmin=min_value, vmax=max_value, cmap=colormap)

    @property
    def min_value(self) -> float | NDArray | None:
        return self._colorize.vmin

    @property
    def max_value(self) -> float | NDArray | None:
        return self._colorize.vmax

    @typing.overload
    def __call__(
        self,
        depth: NDArray,
        dtype: type[np.uint8] = ...,
    ) -> NDArray[np.uint8]: ...  # noqa: GR005 -- preserve positional and keyword API calls

    @typing.overload
    def __call__(
        self,
        depth: NDArray,
        dtype: type[np.float32],
    ) -> NDArray[np.float32]: ...  # noqa: GR005 -- preserve positional and keyword API calls

    @typing.overload
    def __call__(
        self,
        depth: NDArray,
        dtype: type[np.float64],
    ) -> NDArray[np.float64]: ...  # noqa: GR005 -- preserve positional and keyword API calls

    @typing.overload
    def __call__(
        self,
        depth: NDArray,
        dtype: type[np.floating],
    ) -> NDArray[np.floating]: ...  # noqa: GR005 -- preserve positional and keyword API calls

    def __call__(
        self, depth: NDArray, dtype: type[np.uint8] | type[np.floating] = np.uint8
    ) -> NDArray[np.uint8] | NDArray[np.floating]:  # noqa: GR005 -- preserve positional and keyword API calls
        return self._colorize(depth, dtype=dtype)


@typing.overload
def depth2rgb(
    depth: NDArray,
    min_value: float | NDArray | None = ...,
    max_value: float | NDArray | None = ...,
    colormap: str | Callable[[NDArray], NDArray] = ...,
    dtype: type[np.uint8] = ...,
) -> NDArray[np.uint8]: ...  # noqa: GR005 -- preserve positional and keyword API calls


@typing.overload
def depth2rgb(
    depth: NDArray,
    min_value: float | NDArray | None = ...,
    max_value: float | NDArray | None = ...,
    colormap: str | Callable[[NDArray], NDArray] = ...,
    dtype: type[np.float32] = ...,
) -> NDArray[np.float32]: ...  # noqa: GR005 -- preserve positional and keyword API calls


@typing.overload
def depth2rgb(
    depth: NDArray,
    min_value: float | NDArray | None = ...,
    max_value: float | NDArray | None = ...,
    colormap: str | Callable[[NDArray], NDArray] = ...,
    dtype: type[np.float64] = ...,
) -> NDArray[np.float64]: ...  # noqa: GR005 -- preserve positional and keyword API calls


@typing.overload
def depth2rgb(
    depth: NDArray,
    min_value: float | NDArray | None = ...,
    max_value: float | NDArray | None = ...,
    colormap: str | Callable[[NDArray], NDArray] = ...,
    dtype: type[np.floating] = ...,
) -> NDArray[np.floating]: ...  # noqa: GR005 -- preserve positional and keyword API calls


def depth2rgb(
    depth: NDArray,
    min_value: float | NDArray | None = None,
    max_value: float | NDArray | None = None,
    colormap: str | Callable[[NDArray], NDArray] = "jet",
    dtype: type[np.uint8] | type[np.floating] = np.uint8,
) -> NDArray[np.uint8] | NDArray[np.floating]:  # noqa: GR005 -- preserve positional and keyword API calls
    """Deprecated. Use :func:`imgviz.colorize` instead."""
    warnings.warn(
        _DEPRECATION_MESSAGE.format(name="depth2rgb", replacement="colorize"),
        DeprecationWarning,
        stacklevel=2,
    )
    return colorize(depth, vmin=min_value, vmax=max_value, cmap=colormap, dtype=dtype)
