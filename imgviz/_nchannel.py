from __future__ import annotations

import typing
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._normalize import normalize

if TYPE_CHECKING:
    import sklearn.decomposition


class Nchannel2Rgb:
    """Convert nchannel array to rgb by PCA.

    Args:
        pca: PCA object from sklearn.
    """

    def __init__(self, pca: sklearn.decomposition.PCA | None = None) -> None:
        self._pca = pca
        # for uint8
        self._min_max_value: tuple[Any, Any] = (None, None)

    @property
    def pca(self) -> sklearn.decomposition.PCA | None:
        """PCA for N channel to 3."""
        return self._pca

    def __call__(
        self,
        nchannel: NDArray,
        *,
        dtype: type[np.uint8] | type[np.floating] = np.uint8,
    ) -> NDArray[np.uint8] | NDArray[np.floating]:
        """Convert nchannel array to rgb by PCA.

        Args:
            nchannel: N channel image with shape (H, W, C).
            dtype: Output dtype.

        Returns:
            Visualized image with shape (H, W, 3).
        """
        try:
            import sklearn.decomposition
        except ImportError:
            raise ImportError(
                "sklearn is required for Nchannel2Rgb. "
                "Please install scikit-learn or use: pip install imgviz[all]"
            ) from None

        if nchannel.ndim != 3:
            raise ValueError(f"nchannel.ndim must be 3, but got {nchannel.ndim}")
        if not np.issubdtype(nchannel.dtype, np.floating):
            raise ValueError(
                f"nchannel.dtype must be floating, but got {nchannel.dtype}"
            )
        H, W, D = nchannel.shape

        dst = nchannel.reshape(-1, D)
        if self._pca is None:
            self._pca = sklearn.decomposition.PCA(n_components=3, random_state=1234)
            dst = self._pca.fit_transform(dst)
        else:
            dst = self._pca.transform(dst)
        dst = dst.reshape(H, W, 3)

        if dtype == np.uint8:
            if self._min_max_value == (None, None):
                self._min_max_value = (
                    np.nanmin(dst, axis=(0, 1)),
                    np.nanmax(dst, axis=(0, 1)),
                )
            min_value, max_value = self._min_max_value
            dst = normalize(dst, min_value, max_value)
            dst = (dst * 255).round().astype(np.uint8)
        else:
            if not np.issubdtype(dtype, np.floating):
                raise ValueError(f"dtype must be floating, but got {dtype}")
            dst = dst.astype(dtype)

        return dst


@typing.overload
def nchannel2rgb(
    nchannel: NDArray,
    pca: sklearn.decomposition.PCA | None = ...,
    *,
    dtype: type[np.uint8] = ...,
) -> NDArray[np.uint8]: ...


@typing.overload
def nchannel2rgb(
    nchannel: NDArray,
    pca: sklearn.decomposition.PCA | None = ...,
    *,
    dtype: type[np.floating] = ...,
) -> NDArray[np.floating]: ...


def nchannel2rgb(
    nchannel: NDArray,
    pca: sklearn.decomposition.PCA | None = None,
    *,
    dtype: type[np.uint8] | type[np.floating] = np.uint8,
) -> NDArray[np.uint8] | NDArray[np.floating]:
    """Convert nchannel array to rgb by PCA.

    Args:
        nchannel: N channel image with shape (H, W, C).
        pca: PCA object from sklearn.
        dtype: Output dtype.

    Returns:
        Visualized image with shape (H, W, 3).
    """
    return Nchannel2Rgb(pca=pca)(nchannel=nchannel, dtype=dtype)
