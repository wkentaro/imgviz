import numpy as np


def bool2ubyte(img):
    assert img.dtype == bool, "img dtype must be bool"
    return img.astype(np.uint8) * 255


def float2ubyte(img):
    assert np.issubdtype(img.dtype, float), "img dtype must be float"
    assert img.min() >= 0, "img.min() must be >= 0"
    assert img.max() <= 1, "img.max() must be <= 1"
    return (img * 255).round().astype(np.uint8)
