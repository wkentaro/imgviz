import matplotlib.cm
import numpy as np

from .normalize import normalize


class Depth2RGB(object):

    """Convert depth array to rgb."""

    def __init__(self, min_value=None, max_value=None, colormap='jet'):
        self._min_value = min_value
        self._max_value = max_value

        assert hasattr(matplotlib.cm, colormap), \
            'unsupported colormap: {}'.format(colormap)
        self._colormap = colormap

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    def __call__(self, depth, dtype=np.uint8):
        assert depth.ndim == 2, 'depth image must be 2 dimensional'
        assert np.issubdtype(depth.dtype, np.floating), \
            'depth dtype must be float'

        normalized = normalize(
            depth, min_value=self._min_value, max_value=self._max_value
        )

        isnan = np.isnan(normalized)
        normalized[isnan] = 0

        colormap_func = getattr(matplotlib.cm, self._colormap)
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
    dtype=np.uint8,
    min_value=None,
    max_value=None,
    colormap='jet',
):
    '''Convert depth to rgb.

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
    '''
    return Depth2RGB(min_value, max_value, colormap)(depth, dtype)
