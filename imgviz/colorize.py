import warnings

import matplotlib.cm
import numpy as np


def depth2rgb(
    depth,
    min_value=None,
    max_value=None,
    colormap='jet',
):
    if min_value is None:
        min_value = np.nanmin(depth)
    if max_value is None:
        max_value = np.nanmax(depth)

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
