import numpy as np
import PIL.Image

from . import utils

try:
    import cv2
except ImportError:
    cv2 = None


def _resize_pillow(src, height, width, interpolation):
    if interpolation == "linear":
        interpolation = PIL.Image.LINEAR
    elif interpolation == "nearest":
        interpolation = PIL.Image.NEAREST
    else:
        raise ValueError("unsupported interpolation: {}".format(interpolation))

    if np.issubdtype(src.dtype, np.integer):
        dst = utils.numpy_to_pillow(src)
        dst = dst.resize((width, height), resample=interpolation)
        dst = utils.pillow_to_numpy(dst)
    else:
        assert np.issubdtype(src.dtype, np.floating)
        ndim = src.ndim
        if ndim == 2:
            src = src[:, :, None]

        C = src.shape[2]
        dst = np.zeros((height, width, C), dtype=src.dtype)
        for c in range(C):
            src_c = src[:, :, c]
            src_c = utils.numpy_to_pillow(src_c)
            dst[:, :, c] = src_c.resize(
                (width, height), resample=interpolation
            )

        if ndim == 2:
            dst = dst[:, :, 0]
    return dst


def _resize_opencv(src, height, width, interpolation):
    if interpolation == "linear":
        interpolation = cv2.INTER_LINEAR
    elif interpolation == "nearest":
        interpolation = cv2.INTER_NEAREST
    else:
        raise ValueError("unsupported interpolation: {}".format(interpolation))

    dst = cv2.resize(src, (width, height), interpolation=interpolation)
    return dst


def resize(
    src,
    height=None,
    width=None,
    interpolation="linear",
    backend="auto",
):
    """Resize image.

    Parameters
    ----------
    src: numpy.ndarray, (H, W) or (H, W, C)
        Input image.
    height: int, optional
        Height of image. If not given,
        the image is resized based on width keeping image ratio.
    width: int, optional
        Width of image. If not given,
        the image is resized based on height keeping image ratio.
    interpolation: str
        Resizing interpolation (default: 'linear').

        'linear':
            Linear interpolation.
        'nearest':
            Interpolate with the nearest value.

    backend: str
        Resizing backend (default: 'auto').

        'pillow':
            Pillow is used.
        'opencv':
            OpenCV is used.

    Returns
    -------
    dst: numpy.ndarray
        Resized image.

    """
    if not isinstance(src, np.ndarray):
        raise TypeError("src type must be numpy.ndarray")

    if backend == "auto":
        backend = "pillow" if cv2 is None else "opencv"

    src_height, src_width = src.shape[:2]
    if isinstance(width, float):
        scale_width = width
        width = int(round(scale_width * src_width))
    if isinstance(height, float):
        scale_height = height
        height = int(round(scale_height * src_height))
    if height is None:
        assert width is not None
        scale_height = 1.0 * width / src_width
        height = int(round(scale_height * src_height))
    if width is None:
        assert height is not None
        scale_width = 1.0 * height / src_height
        width = int(round(scale_width * src_width))

    if backend == "pillow":
        dst = _resize_pillow(src, height, width, interpolation)
    elif backend == "opencv":
        dst = _resize_opencv(src, height, width, interpolation)
    else:
        raise ValueError("unsupported backend: {}".format(backend))

    return dst
