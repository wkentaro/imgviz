from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._colorize import colorize


def _to_scalar(image: NDArray) -> NDArray[np.float64]:
    arr = image.astype(np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        red, green, blue = 0.299, 0.587, 0.114
        return red * arr[:, :, 0] + green * arr[:, :, 1] + blue * arr[:, :, 2]
    raise ValueError(
        f"image must be (H, W), (H, W, 3) or (H, W, 4), but got shape {image.shape}"
    )


def diff(
    a: NDArray,
    b: NDArray,
    mode: Literal["signed", "abs", "ssim"] = "signed",
    vmin: float | None = None,
    vmax: float | None = None,
) -> NDArray[np.uint8]:
    """Visualize the difference between two images.

    Color images are reduced to luminance before differencing, so the result
    is always a colorized scalar field with shape (H, W, 3).

    Args:
        a: First image with shape (H, W), (H, W, 3) or (H, W, 4).
        b: Second image with the same shape as ``a``.
        mode: ``"signed"`` maps ``a - b`` onto a diverging colormap centered at
            zero, ``"abs"`` maps ``|a - b|`` onto a sequential colormap, and
            ``"ssim"`` colorizes the local SSIM map (requires scikit-image).
        vmin: Lower bound for normalization. Defaults are mode-specific: a
            symmetric bound for ``"signed"``, ``0`` for ``"abs"``, and the
            data minimum for ``"ssim"``.
        vmax: Upper bound for normalization. Defaults to the data range.

    Returns:
        Colorized difference image with shape (H, W, 3) and dtype ``uint8``.

    Examples:
        >>> a = imgviz.data.arc2017()["rgb"]
        >>> b = a.copy()
        >>> b[:50, :50] = 0
        >>> signed = imgviz.diff(a, b, mode="signed")
        >>> magnitude = imgviz.diff(a, b, mode="abs")
        >>> structural = imgviz.diff(a, b, mode="ssim")
    """
    if a.shape != b.shape:
        raise ValueError(
            f"a and b must have the same shape, but got {a.shape} and {b.shape}"
        )

    scalar_a = _to_scalar(a)
    scalar_b = _to_scalar(b)

    if mode == "signed":
        signed = scalar_a - scalar_b
        if vmin is not None and vmax is not None:
            return colorize(signed, vmin=vmin, vmax=vmax, cmap="coolwarm")
        if vmin is None and vmax is None:
            extent = float(np.nanmax(np.abs(signed)))
        elif vmin is None:
            assert vmax is not None
            extent = abs(vmax)
        else:
            extent = abs(vmin)
        return colorize(signed, vmin=-extent, vmax=extent, cmap="coolwarm")

    if mode == "abs":
        magnitude = np.abs(scalar_a - scalar_b)
        if vmin is None:
            vmin = 0.0
        return colorize(magnitude, vmin=vmin, vmax=vmax, cmap="magma")

    if mode == "ssim":
        try:
            import skimage.metrics
        except ImportError:
            raise ImportError(
                "skimage is required for mode='ssim'. "
                "Please install scikit-image or use: pip install imgviz[all]"
            ) from None

        data_max = max(scalar_a.max(), scalar_b.max())
        data_min = min(scalar_a.min(), scalar_b.min())
        data_range = float(data_max - data_min) or 1.0
        _, similarity = skimage.metrics.structural_similarity(
            scalar_a, scalar_b, data_range=data_range, full=True
        )
        return colorize(similarity, vmin=vmin, vmax=vmax, cmap="viridis")

    raise ValueError(f"mode must be 'signed', 'abs' or 'ssim', but got {mode!r}")
