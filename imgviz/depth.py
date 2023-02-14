import typing  # NOQA

import matplotlib
import numpy as np

from .normalize import normalize


class Depth2RGB(object):

    """Convert depth array to rgb.

    Parameters
    ----------
    min_value: float, optional
        Minimum value for colorizing.
    max_value: float, optional
        Maximum value for colorizing.
    colormap: str, optional
        Colormap, default: 'jet'.

    """

    def __init__(self, min_value=None, max_value=None, colormap="jet"):
        self._min_value = min_value
        self._max_value = max_value
        self._colormap = colormap

    @property
    def min_value(self):
        """Minimum value of depth."""
        return self._min_value

    @property
    def max_value(self):
        """Maximum value of depth."""
        return self._max_value

    def __call__(self, depth, dtype=np.uint8):
        """Convert depth array to rgb.

        Parameters
        ----------
        depth: numpy.ndarray, (H, W), float
            Depth image.
        dtype: numpy.dtype
            Dtype of output image. default: np.uint8

        Returns
        -------
        rgb: numpy.ndarray, (H, W, 3), np.uint8
            Output colorized image.

        """
        assert depth.ndim == 2, "depth image must be 2 dimensional"
        assert np.issubdtype(
            depth.dtype, np.floating
        ), "depth dtype must be float"

        normalized, self._min_value, self._max_value = normalize(
            depth,
            min_value=self._min_value,
            max_value=self._max_value,
            return_minmax=True,
        )

        isnan = np.isnan(normalized)
        normalized[isnan] = 0

        if isinstance(self._colormap, str):
            colormap_func = matplotlib.colormaps[self._colormap]
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
    depth,
    min_value=None,
    max_value=None,
    colormap="jet",
    dtype=np.uint8,
):
    # type: (np.ndarray, typing.Optional[float], typing.Optional[float], str, typing.Type) -> np.ndarray  # NOQA
    """Convert depth to rgb.

    Parameters
    ----------
    depth: numpy.ndarray, (H, W), float
        Depth image.
    dtype: numpy.dtype
        Dtype of output image. default: np.uint8
    min_value: float, optional
        Minimum value for colorizing.
    max_value: float, optional
        Maximum value for colorizing.
    colormap: str, optional
        Colormap, default: 'jet'.

    Returns
    -------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Output colorized image.

    """
    return Depth2RGB(min_value, max_value, colormap)(depth, dtype)
