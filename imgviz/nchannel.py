import numpy as np

from . import normalize


class Nchannel2RGB(object):
    """Convert nchannel array to rgb by PCA.

    Parameters
    ----------
    pca: sklearn.decomposition.PCA
        PCA.

    """

    def __init__(self, pca=None):
        self._pca = pca
        # for uint8
        self._min_max_value = (None, None)

    @property
    def pca(self):
        """PCA for N channel to 3."""
        return self._pca

    def __call__(self, nchannel, dtype=np.uint8):
        """Convert nchannel array to rgb by PCA.

        Parameters
        ----------
        nchannel: numpy.ndarray, (H, W, C), float
            N channel image.
        dtype: numpy.dtype
            Dtype (default: numpy.uint8).

        Returns
        -------
        dst: numpy.ndarray, (H, W, 3), numpy.uint8
            Visualized image.

        """
        import sklearn.decomposition

        assert nchannel.ndim == 3, "nchannel.ndim must be 3"
        assert np.issubdtype(
            nchannel.dtype, np.floating
        ), "nchannel.dtype must be floating"
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
            dst = normalize.normalize(dst, min_value, max_value)
            dst = (dst * 255).round().astype(np.uint8)
        else:
            assert np.issubdtype(dtype, np.floating)
            dst = dst.astype(dtype)

        return dst


def nchannel2rgb(nchannel, dtype=np.uint8, pca=None):
    """Convert nchannel array to rgb by PCA.

    Parameters
    ----------
    nchannel: numpy.ndarray, (H, W, C), float
        N channel image.
    dtype: numpy.dtype
        Dtype (default: numpy.uint8).
    pca: sklearn.decomposition.PCA
        PCA.

    Returns
    -------
    dst: numpy.ndarray, (H, W, 3), numpy.uint8
        Visualized image.

    """
    return Nchannel2RGB(pca)(nchannel, dtype)
