import numpy as np

from . import dtype
from . import utils


def rgb2gray(rgb):
    # type: (np.ndarray) -> np.ndarray
    """Covnert rgb to gray.

    Parameters
    ----------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Input rgb image.

    Returns
    -------
    gray: numpy.ndarray, (H, W)
        Output gray image.

    """
    assert rgb.ndim == 3, "rgb must be 3 dimensional"
    assert rgb.shape[2] == 3, "rgb shape must be (H, W, 3)"
    assert rgb.dtype == np.uint8, "rgb dtype must be np.uint8"

    gray = utils.numpy_to_pillow(rgb)
    gray = gray.convert("L")
    gray = utils.pillow_to_numpy(gray)
    return gray


def gray2rgb(gray):
    # type: (np.ndarray) -> np.ndarray
    """Covnert gray to rgb.

    Parameters
    ----------
    gray: numpy.ndarray, (H, W), np.uint8
        Input gray image.

    Returns
    -------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Output rgb image.

    """
    assert gray.ndim == 2, "gray must be 2 dimensional"
    assert gray.dtype == np.uint8, "gray dtype must be np.uint8"

    rgb = gray[:, :, None].repeat(3, axis=2)
    return rgb


def rgb2rgba(rgb):
    # type: (np.ndarray) -> np.ndarray
    """Convert rgb to rgba.

    Parameters
    ----------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Input rgb image.

    Returns
    -------
    rgba: numpy.ndarray, (H, W, 4), np.uint8
        Output rgba image.

    """
    assert rgb.ndim == 3, "rgb must be 3 dimensional"
    assert rgb.shape[2] == 3, "rgb shape must be (H, W, 3)"
    assert rgb.dtype == np.uint8, "rgb dtype must be np.uint8"

    a = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack((rgb, a))
    return rgba


def rgb2hsv(rgb):
    # type: (np.ndarray) -> np.ndarray
    """Convert rgb to hsv.

    Parameters
    ----------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Input rgb image.

    Returns
    -------
    hsv: numpy.ndarray, (H, W, 3), np.uint8
        Output hsv image.

    """
    hsv = utils.numpy_to_pillow(rgb, mode="RGB")
    hsv = hsv.convert("HSV")
    hsv = utils.pillow_to_numpy(hsv)
    return hsv


def hsv2rgb(hsv):
    # type: (np.ndarray) -> np.ndarray
    """Convert hsv to rgb.

    Parameters
    ----------
    hsv: numpy.ndarray, (H, W, 3), np.uint8
        Input hsv image.

    Returns
    -------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Output rgb image.

    """
    rgb = utils.numpy_to_pillow(hsv, mode="HSV")
    rgb = rgb.convert("RGB")
    rgb = utils.pillow_to_numpy(rgb)
    return rgb


def rgba2rgb(rgba):
    # type: (np.ndarray) -> np.ndarray
    """Convert rgba to rgb.

    Parameters
    ----------
    rgba: numpy.ndarray, (H, W, 4), np.uint8
        Input rgba image.

    Returns
    -------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Output rgb image.
    """
    rgb = rgba[:, :, :3]
    return rgb


def asgray(img):
    # type: (np.ndarray) -> np.ndarray
    """Convert any array to gray image.

    Parameters
    ----------
    img: numpy.ndarray
        Input image.

    Returns
    -------
    gray: numpy.ndarray, (H, W), np.uint8
        Output gray image.
    """
    if img.ndim == 2:
        if img.dtype == bool:
            gray = dtype.bool2ubyte(img)
        else:
            gray = img
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = rgb2gray(rgba2rgb(img))
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = rgb2gray(img)
    else:
        raise ValueError(
            "Unsupported image format to convert to gray:" "shape={}, dtype={}".format(
                img.shape, img.dtype
            )
        )
    return gray


def asrgb(img):
    # type: (np.ndarray) -> np.ndarray
    """Convert any array to rgb image.

    Parameters
    ----------
    img: numpy.ndarray
        Input image.

    Returns
    -------
    rgb: numpy.ndarray, (H, W, 3), np.uint8
        Output gray image.
    """
    if img.ndim == 2:
        if img.dtype == bool:
            img = dtype.bool2ubyte(img)
        rgb = gray2rgb(img)
    elif img.ndim == 3 and img.shape[2] == 4:
        rgb = rgba2rgb(img)
    elif img.ndim == 3 and img.shape[2] == 3:
        rgb = img
    else:
        raise ValueError(
            "Unsupported image format to convert to rgb:" "shape={}, dtype={}".format(
                img.shape, img.dtype
            )
        )
    return rgb


def asrgba(img):
    # type: (np.ndarray) -> np.ndarray
    """Convert any array to rgba image.

    Parameters
    ----------
    img: numpy.ndarray
        Input image.

    Returns
    -------
    rgba: numpy.ndarray, (H, W, 3), np.uint8
        Output gray image.
    """
    if img.ndim == 2:
        if img.dtype == bool:
            img = dtype.bool2ubyte(img)
        rgb = gray2rgb(img)
        rgba = rgb2rgba(rgb)
    if img.ndim == 3 and img.shape[2] == 4:
        rgba = img
    elif img.ndim == 3 and img.shape[2] == 3:
        rgba = rgb2rgba(img)
    else:
        raise ValueError(
            "Unsupported image format to convert to rgba:" "shape={}, dtype={}".format(
                img.shape, img.dtype
            )
        )
    return rgba


def get_fg_color(color):
    color = np.asarray(color, dtype=np.uint8)
    intensity = rgb2gray(color.reshape(1, 1, 3)).sum()
    if intensity > 170:
        return (0, 0, 0)
    return (255, 255, 255)
