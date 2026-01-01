from __future__ import annotations

from collections.abc import Callable

import cmap
import numpy as np
from numpy.typing import DTypeLike
from numpy.typing import NDArray

from .normalize import normalize


class Depth2RGB:
    """Convert depth array to rgb.

    Parameters
    ----------
    min_value
        Minimum value for colorizing.
    max_value
        Maximum value for colorizing.
    colormap
        Colormap name or callable.

    """

    def __init__(
        self,
        min_value: float | NDArray | None = None,
        max_value: float | NDArray | None = None,
        colormap: str | Callable[[NDArray], NDArray] = "jet",
    ) -> None:
        self._min_value = min_value
        self._max_value = max_value
        self._colormap = colormap

    @property
    def min_value(self) -> float | NDArray | None:
        """Minimum value of depth."""
        return self._min_value

    @property
    def max_value(self) -> float | NDArray | None:
        """Maximum value of depth."""
        return self._max_value

    def __call__(
        self, depth: NDArray, dtype: DTypeLike = np.uint8
    ) -> NDArray[np.uint8] | NDArray[np.floating]:
        """Convert depth array to rgb.

        Parameters
        ----------
        depth
            Depth image with shape (H, W).
        dtype
            Output dtype.

        Returns
        -------
        rgb
            Colorized image with shape (H, W, 3).

        """
        assert depth.ndim == 2, "depth image must be 2 dimensional"
        assert np.issubdtype(depth.dtype, np.floating), "depth dtype must be float"

        normalized, self._min_value, self._max_value = normalize(
            depth,
            min_value=self._min_value,
            max_value=self._max_value,
            return_minmax=True,
        )

        isnan = np.isnan(normalized)
        normalized[isnan] = 0

        if isinstance(self._colormap, str):
            colormap_func = cmap.Colormap(self._colormap)
        else:
            colormap_func = self._colormap
        rgb = colormap_func(normalized)[:, :, :3]
        rgb[isnan] = (0, 0, 0)

        if dtype == np.uint8:
            rgb = (rgb * 255).round().astype(np.uint8)
        else:
            assert np.issubdtype(dtype, np.floating)
            rgb = rgb.astype(dtype)

        return rgb


def depth2rgb(
    depth: NDArray,
    min_value: float | NDArray | None = None,
    max_value: float | NDArray | None = None,
    colormap: str | Callable[[NDArray], NDArray] = "jet",
    dtype: DTypeLike = np.uint8,
) -> NDArray[np.uint8] | NDArray[np.floating]:
    """Convert depth to rgb.

    Parameters
    ----------
    depth
        Depth image with shape (H, W).
    min_value
        Minimum value for colorizing.
    max_value
        Maximum value for colorizing.
    colormap
        Colormap name or callable.
    dtype
        Output dtype.

    Returns
    -------
    rgb
        Colorized image with shape (H, W, 3).

    """
    return Depth2RGB(min_value, max_value, colormap)(depth, dtype)
