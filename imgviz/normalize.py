import numbers
import warnings

import numpy as np


def normalize(src, min_value=None, max_value=None):
    if min_value is None:
        min_value = np.nanmin(src)
    else:
        assert isinstance(min_value, numbers.Real), \
            'min_value must be float type'

    if max_value is None:
        max_value = np.nanmax(src)
    else:
        assert isinstance(max_value, numbers.Real), \
            'max_value must be float type'

    if np.isinf(min_value) or np.isinf(max_value):
        warnings.warn('min or max value of input array is inf.')

    if max_value == min_value:
        eps = np.finfo(src.dtype).eps
        max_value += eps
        min_value -= eps

    dst = np.zeros(src.shape, dtype=float)

    isnan = np.isnan(src)
    dst[~isnan] = (
        1. * (src[~isnan] - min_value) / (max_value - min_value)
    )
    dst[isnan] = np.nan

    return dst
