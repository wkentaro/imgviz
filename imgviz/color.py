import numpy as np
import PIL.Image


def rgb2gray(rgb):
    # type: (np.ndarray) -> np.ndarray
    assert rgb.ndim == 3
    gray = PIL.Image.fromarray(rgb)
    gray = gray.convert('L')
    gray = np.asarray(gray)
    return gray


def gray2rgb(gray):
    # type: (np.ndarray) -> np.ndarray
    assert gray.ndim == 2
    rgb = gray[:, :, None].repeat(3, axis=2)
    return rgb


def rgb2rgba(rgb):
    # type: (np.ndarray) -> np.ndarray
    assert rgb.ndim == 3
    a = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack((rgb, a))
    return rgba
