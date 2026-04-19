from __future__ import annotations

import typing
from collections.abc import Callable

import cmap as _cmap
import numpy as np
from numpy.typing import NDArray

from ._normalize import normalize


class Colorize:
    """Apply a colormap to a 2D scalar field.

    Args:
        vmin: Minimum value for normalization. If ``None``, it is computed
            from the first call's input and cached on the instance; the
            same value is then reused for every subsequent call.
        vmax: Maximum value for normalization. Same caching semantics as
            ``vmin``.
        cmap: Colormap name or callable mapping values in [0, 1] to RGBA.
            A string name is resolved once at construction time.
    """

    def __init__(
        self,
        vmin: float | NDArray | None = None,
        vmax: float | NDArray | None = None,
        cmap: str | Callable[[NDArray], NDArray] = "viridis",
    ) -> None:
        self._vmin = vmin
        self._vmax = vmax
        self._cmap_func: Callable[[NDArray], NDArray] = (
            _cmap.Colormap(cmap) if isinstance(cmap, str) else cmap
        )

    @property
    def vmin(self) -> float | NDArray | None:
        return self._vmin

    @property
    def vmax(self) -> float | NDArray | None:
        return self._vmax

    @typing.overload
    def __call__(
        self,
        scalar: NDArray,
        dtype: type[np.uint8] = ...,
    ) -> NDArray[np.uint8]: ...

    @typing.overload
    def __call__(
        self,
        scalar: NDArray,
        dtype: type[np.float32],
    ) -> NDArray[np.float32]: ...

    @typing.overload
    def __call__(
        self,
        scalar: NDArray,
        dtype: type[np.float64],
    ) -> NDArray[np.float64]: ...

    @typing.overload
    def __call__(
        self,
        scalar: NDArray,
        dtype: type[np.floating],
    ) -> NDArray[np.floating]: ...

    def __call__(
        self, scalar: NDArray, dtype: type[np.uint8] | type[np.floating] = np.uint8
    ) -> NDArray[np.uint8] | NDArray[np.floating]:
        if scalar.ndim != 2:
            raise ValueError(f"scalar must be 2 dimensional, but got {scalar.ndim}")
        if not np.issubdtype(scalar.dtype, np.floating):
            raise ValueError(f"scalar dtype must be float, but got {scalar.dtype}")

        normalized, auto_vmin, auto_vmax = normalize(
            scalar,
            min_value=self._vmin,
            max_value=self._vmax,
            return_minmax=True,
        )
        if self._vmin is None:
            self._vmin = auto_vmin
        if self._vmax is None:
            self._vmax = auto_vmax

        isnan = np.isnan(normalized)
        normalized[isnan] = 0

        rgb = self._cmap_func(normalized)[:, :, :3]
        rgb[isnan] = (0, 0, 0)

        if dtype == np.uint8:
            rgb = (rgb * 255).round().astype(np.uint8)
        else:
            if not np.issubdtype(dtype, np.floating):
                raise ValueError(
                    f"dtype must be np.uint8 or a floating type, but got {dtype}"
                )
            rgb = rgb.astype(dtype)

        return rgb


@typing.overload
def colorize(
    scalar: NDArray,
    vmin: float | NDArray | None = ...,
    vmax: float | NDArray | None = ...,
    cmap: str | Callable[[NDArray], NDArray] = ...,
    dtype: type[np.uint8] = ...,
) -> NDArray[np.uint8]: ...


@typing.overload
def colorize(
    scalar: NDArray,
    vmin: float | NDArray | None = ...,
    vmax: float | NDArray | None = ...,
    cmap: str | Callable[[NDArray], NDArray] = ...,
    dtype: type[np.float32] = ...,
) -> NDArray[np.float32]: ...


@typing.overload
def colorize(
    scalar: NDArray,
    vmin: float | NDArray | None = ...,
    vmax: float | NDArray | None = ...,
    cmap: str | Callable[[NDArray], NDArray] = ...,
    dtype: type[np.float64] = ...,
) -> NDArray[np.float64]: ...


@typing.overload
def colorize(
    scalar: NDArray,
    vmin: float | NDArray | None = ...,
    vmax: float | NDArray | None = ...,
    cmap: str | Callable[[NDArray], NDArray] = ...,
    dtype: type[np.floating] = ...,
) -> NDArray[np.floating]: ...


def colorize(
    scalar: NDArray,
    vmin: float | NDArray | None = None,
    vmax: float | NDArray | None = None,
    cmap: str | Callable[[NDArray], NDArray] = "viridis",
    dtype: type[np.uint8] | type[np.floating] = np.uint8,
) -> NDArray[np.uint8] | NDArray[np.floating]:
    """Apply a colormap to a 2D scalar field.

    Works for any 2D scalar field: depth maps, attention maps, heatmaps,
    score fields, single-channel model outputs.

    Args:
        scalar: 2D scalar field with shape (H, W).
        vmin: Minimum value for normalization.
        vmax: Maximum value for normalization.
        cmap: Colormap name or callable mapping values in [0, 1] to RGBA.
        dtype: Output dtype.

    Returns:
        Colorized image with shape (H, W, 3).
    """
    return Colorize(vmin=vmin, vmax=vmax, cmap=cmap)(scalar, dtype=dtype)
