import numpy as np
import PIL.Image


def rgb2gray(rgb):
    # type: (np.ndarray) -> np.ndarray
    '''Covnert rgb to gray.

    Parameters
    ----------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Input rgb image.

    Returns
    -------
    gray: numpy.ndarray, (H, W)
        Output gray image.
    '''
    assert rgb.ndim == 3, 'rgb must be 3 dimensional'
    assert rgb.shape[2] == 3, 'rgb shape must be (H, W, 3)'
    assert rgb.dtype == np.uint8, 'rgb dtype must be np.uint8'

    gray = PIL.Image.fromarray(rgb)
    gray = gray.convert('L')
    gray = np.asarray(gray)
    return gray


def gray2rgb(gray):
    # type: (np.ndarray) -> np.ndarray
    '''Covnert gray to rgb.

    Parameters
    ----------
    gray: numpy.ndarray, (H, W), np.uint8
        Input gray image.

    Returns
    -------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Output rgb image.
    '''
    assert gray.ndim == 2, 'gray must be 2 dimensional'
    assert gray.dtype == np.uint8, 'gray dtype must be np.uint8'

    rgb = gray[:, :, None].repeat(3, axis=2)
    return rgb


def rgb2rgba(rgb):
    # type: (np.ndarray) -> np.ndarray
    '''Convert rgb to rgba.

    Parameters
    ----------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Input rgb image.

    Returns
    -------
    rgba: numpy.ndarray, (H, W, 4), np.uint8
        Output rgba image.
    '''
    assert rgb.ndim == 3, 'rgb must be 3 dimensional'
    assert rgb.shape[2] == 3, 'rgb shape must be (H, W, 3)'
    assert rgb.dtype == np.uint8, 'rgb dtype must be np.uint8'

    a = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack((rgb, a))
    return rgba


def rgb2hsv(rgb):
    hsv = PIL.Image.fromarray(rgb, mode='RGB')
    hsv = hsv.convert('HSV')
    hsv = np.array(hsv)
    return hsv


def hsv2rgb(hsv):
    rgb = PIL.Image.fromarray(hsv, mode='HSV')
    rgb = rgb.convert('RGB')
    rgb = np.array(rgb)
    return rgb


def get_fg_color(color):
    color = np.asarray(color, dtype=np.uint8)
    intensity = rgb2gray(color.reshape(1, 1, 3)).sum()
    if intensity > 170:
        return (0, 0, 0)
    return (255, 255, 255)
