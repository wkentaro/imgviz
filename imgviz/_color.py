from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import _dtype
from . import _utils


def rgb2gray(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert rgb to gray.

    Args:
        rgb: Input rgb image with shape (H, W, 3).

    Returns:
        Gray image with shape (H, W).
    """
    if rgb.ndim != 3:
        raise ValueError(f"rgb must be 3 dimensional, but got {rgb.ndim}")
    if rgb.shape[2] != 3:
        raise ValueError(f"rgb shape must be (H, W, 3), but got {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb dtype must be np.uint8, but got {rgb.dtype}")

    gray = _utils.numpy_to_pillow(rgb)
    gray = gray.convert("L")
    gray = _utils.pillow_to_numpy(gray)
    return gray


def gray2rgb(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert gray to rgb.

    Args:
        gray: Input gray image with shape (H, W).

    Returns:
        RGB image with shape (H, W, 3).
    """
    if gray.ndim != 2:
        raise ValueError(f"gray must be 2 dimensional, but got {gray.ndim}")
    if gray.dtype != np.uint8:
        raise ValueError(f"gray dtype must be np.uint8, but got {gray.dtype}")

    rgb = gray[:, :, None].repeat(3, axis=2)
    return rgb


def rgb2rgba(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert rgb to rgba.

    Args:
        rgb: Input rgb image with shape (H, W, 3).

    Returns:
        RGBA image with shape (H, W, 4).
    """
    if rgb.ndim != 3:
        raise ValueError(f"rgb must be 3 dimensional, but got {rgb.ndim}")
    if rgb.shape[2] != 3:
        raise ValueError(f"rgb shape must be (H, W, 3), but got {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb dtype must be np.uint8, but got {rgb.dtype}")

    a = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack((rgb, a))
    return rgba


def rgb2hsv(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert rgb to hsv.

    Args:
        rgb: Input rgb image with shape (H, W, 3).

    Returns:
        HSV image with shape (H, W, 3).
    """
    hsv = _utils.numpy_to_pillow(rgb, mode="RGB")
    hsv = hsv.convert("HSV")
    hsv = _utils.pillow_to_numpy(hsv)
    return hsv


def hsv2rgb(hsv: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert hsv to rgb.

    Args:
        hsv: Input hsv image with shape (H, W, 3).

    Returns:
        RGB image with shape (H, W, 3).
    """
    rgb = _utils.numpy_to_pillow(hsv, mode="HSV")
    rgb = rgb.convert("RGB")
    rgb = _utils.pillow_to_numpy(rgb)
    return rgb


def rgba2rgb(rgba: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert rgba to rgb.

    Args:
        rgba: Input rgba image with shape (H, W, 4).

    Returns:
        RGB image with shape (H, W, 3).
    """
    rgb = rgba[:, :, :3]
    return rgb


def asgray(image: NDArray) -> NDArray[np.uint8]:
    """Convert any array to gray image.

    Args:
        image: Input image.

    Returns:
        Gray image with shape (H, W).
    """
    if image.ndim == 2:
        if image.dtype == bool:
            gray = _dtype.bool2ubyte(image)
        else:
            gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = rgb2gray(rgba2rgb(image))
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = rgb2gray(image)
    else:
        raise ValueError(
            f"unsupported image format to convert to gray: "
            f"shape={image.shape}, dtype={image.dtype}"
        )
    return gray


def asrgb(image: NDArray, copy: bool = False) -> NDArray[np.uint8]:
    """Convert any array to rgb image.

    Args:
        image: Input image.
        copy: Whether to return a copy of the image.

    Returns:
        RGB image with shape (H, W, 3).
    """
    if image.ndim == 2:
        if image.dtype == bool:
            image = _dtype.bool2ubyte(image)
        rgb = gray2rgb(image)
    elif image.ndim == 3 and image.shape[2] == 4:
        rgb = rgba2rgb(image)
    elif image.ndim == 3 and image.shape[2] == 3:
        rgb = image.copy() if copy else image
    else:
        raise ValueError(
            f"unsupported image format to convert to rgb: "
            f"shape={image.shape}, dtype={image.dtype}"
        )
    return rgb


def asrgba(image: NDArray) -> NDArray[np.uint8]:
    """Convert any array to rgba image.

    Args:
        image: Input image.

    Returns:
        RGBA image with shape (H, W, 4).
    """
    if image.ndim == 2:
        if image.dtype == bool:
            image = _dtype.bool2ubyte(image)
        rgb = gray2rgb(image)
        rgba = rgb2rgba(rgb)
    elif image.ndim == 3 and image.shape[2] == 4:
        rgba = image
    elif image.ndim == 3 and image.shape[2] == 3:
        rgba = rgb2rgba(image)
    else:
        raise ValueError(
            f"unsupported image format to convert to rgba: "
            f"shape={image.shape}, dtype={image.dtype}"
        )
    return rgba


def get_fg_color(
    color: tuple[int, int, int] | NDArray[np.uint8],
) -> tuple[int, int, int]:
    """Get foreground color (black or white) for given background color."""
    color_arr: NDArray[np.uint8] = np.asarray(color, dtype=np.uint8)
    intensity: np.uint8 = rgb2gray(color_arr.reshape(1, 1, 3)).sum()
    if intensity > 170:
        return (0, 0, 0)
    return (255, 255, 255)
