import warnings

import matplotlib.cm
import numpy as np
import PIL.Image


def rgb2gray(rgb):
    assert rgb.ndim == 3
    gray = PIL.Image.fromarray(rgb)
    gray = gray.convert('L')
    gray = np.asarray(gray)
    return gray


def gray2rgb(gray):
    assert gray.ndim == 2
    rgb = gray[:, :, None].repeat(3, axis=2)
    return rgb


def rgb2rgba(rgb):
    assert rgb.ndim == 3
    a = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack((rgb, a))
    return rgba


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
