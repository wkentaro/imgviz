import numbers
import warnings

import matplotlib.cm
import numpy as np


def depth2rgb(
    depth,
    min_value=None,
    max_value=None,
    colormap='jet',
):
    '''Convert depth to rgb.

    Parameters
    ----------
    depth: numpy.ndarray, (H, W), float
        Depth image.
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
    assert depth.ndim == 2, 'depth image must be 2 dimensional'
    assert np.issubdtype(depth.dtype, np.floating), 'depth dtype must be float'

    if min_value is None:
        min_value = np.nanmin(depth)
    else:
        assert isinstance(min_value, numbers.Real), \
            'min_value must be float type'

    if max_value is None:
        max_value = np.nanmax(depth)
    else:
        assert isinstance(max_value, numbers.Real), \
            'max_value must be float type'

    assert hasattr(matplotlib.cm, colormap), \
        'unsupported colormap: {}'.format(colormap)

    if np.isinf(min_value) or np.isinf(max_value):
        warnings.warn('Min or max value for depth colorization is inf.')
    if max_value == min_value:
        eps = np.finfo(depth.dtype).eps
        max_value += eps
        min_value -= eps

    rgb = np.zeros(depth.shape, dtype=float)

    isnan = np.isnan(depth)
    rgb[~isnan] = 1. * (depth[~isnan] - min_value) / (max_value - min_value)

    colormap_func = getattr(matplotlib.cm, colormap)
    rgb = colormap_func(rgb)[:, :, :3]
    rgb = (rgb * 255).round().astype(np.uint8)
    rgb[isnan] = (0, 0, 0)

    return rgb
