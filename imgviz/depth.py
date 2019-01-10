import matplotlib.cm
import numpy as np

from .normalize import normalize


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

    assert hasattr(matplotlib.cm, colormap), \
        'unsupported colormap: {}'.format(colormap)

    normalized = normalize(depth, min_value, max_value)

    isnan = np.isnan(normalized)
    normalized[isnan] = 0

    colormap_func = getattr(matplotlib.cm, colormap)
    rgb = colormap_func(normalized)[:, :, :3]
    rgb = (rgb * 255).round().astype(np.uint8)
    rgb[isnan] = (0, 0, 0)

    return rgb
