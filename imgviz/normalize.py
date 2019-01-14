import warnings

import numpy as np


def normalize(src, min_value=None, max_value=None):
    if src.ndim == 2:
        D = 1
    else:
        assert src.ndim == 3, 'src ndim must be 2 or 3'
        D = src.shape[2]

    if min_value is None:
        min_value = np.nanmin(src, axis=(0, 1))
    min_value = np.atleast_1d(min_value).astype(float)
    assert min_value.shape == (D,)

    if max_value is None:
        max_value = np.nanmax(src, axis=(0, 1))
    max_value = np.atleast_1d(max_value).astype(float)
    assert max_value.shape == (D,)

    if np.isinf(min_value).any() or np.isinf(max_value).any():
        warnings.warn('some of min or max values are inf.')

    eps = np.finfo(src.dtype).eps
    issame = max_value == min_value
    min_value[issame] -= eps
    max_value[issame] += eps

    dst = np.zeros(src.shape, dtype=float)

    if src.ndim == 2:
        isnan = np.isnan(src)
    else:
        isnan = np.isnan(src).any(axis=2)
    dst[~isnan] = (
        1. * (src[~isnan] - min_value) / (max_value - min_value)
    )
    dst[isnan] = np.nan

    return dst
