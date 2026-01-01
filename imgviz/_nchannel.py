from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import DTypeLike
from numpy.typing import NDArray

from .normalize import normalize

if TYPE_CHECKING:
    import sklearn.decomposition


class Nchannel2RGB:
    """Convert nchannel array to rgb by PCA.

    Parameters
    ----------
    pca
        PCA object from sklearn.

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
        self, nchannel: NDArray, dtype: DTypeLike = np.uint8
    ) -> NDArray[np.uint8] | NDArray[np.floating]:
        """Convert nchannel array to rgb by PCA.

        Parameters
        ----------
        nchannel
            N channel image with shape (H, W, C).
        dtype
            Output dtype.

        Returns
        -------
        dst
            Visualized image with shape (H, W, 3).

        """
        import sklearn.decomposition

        assert nchannel.ndim == 3, "nchannel.ndim must be 3"
        assert np.issubdtype(nchannel.dtype, np.floating), (
            "nchannel.dtype must be floating"
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
            if self._min_max_value is None:
                self._min_max_value = (
                    np.nanmin(dst, axis=(0, 1)),
                    np.nanmax(dst, axis=(0, 1)),
                )
            min_value, max_value = self._min_max_value
            dst = normalize(dst, min_value, max_value)
            dst = (dst * 255).round().astype(np.uint8)
        else:
            assert np.issubdtype(dtype, np.floating)
            dst = dst.astype(dtype)

        return dst


def nchannel2rgb(
    nchannel: NDArray,
    dtype: DTypeLike = np.uint8,
    pca: sklearn.decomposition.PCA | None = None,
) -> NDArray[np.uint8] | NDArray[np.floating]:
    """Convert nchannel array to rgb by PCA.

    Parameters
    ----------
    nchannel
        N channel image with shape (H, W, C).
    dtype
        Output dtype.
    pca
        PCA object from sklearn.

    Returns
    -------
    dst
        Visualized image with shape (H, W, 3).

    """
    return Nchannel2RGB(pca)(nchannel, dtype)
