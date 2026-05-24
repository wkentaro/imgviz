from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._colorize import colorize


def _to_luminance(image: NDArray) -> NDArray[np.float64]:
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

    Color images are reduced to BT.601 luminance before differencing
    (alpha is ignored for 4-channel inputs), so the result is always a
    colorized scalar field with shape (H, W, 3).

    Args:
        a: First image with shape (H, W), (H, W, 3) or (H, W, 4).
        b: Second image with the same shape as ``a``.
        mode: ``"signed"`` maps ``a - b`` onto a diverging colormap centered at
            zero, ``"abs"`` maps ``|a - b|`` onto a sequential colormap, and
            ``"ssim"`` colorizes the local SSIM map (requires scikit-image).
        vmin: Lower bound for the colormap. Mode-specific defaults:

            - ``"signed"``: if both ``vmin`` and ``vmax`` are ``None``, the
              bound is taken symmetrically from ``|a - b|``. If exactly one
              is given, its magnitude defines the symmetric range. If both
              are given they are used as-is (allowing asymmetric ranges).
            - ``"abs"``: ``0``.
            - ``"ssim"``: the minimum of the local SSIM map. Pass
              ``vmin=-1, vmax=1`` to use the absolute SSIM bounds so two
              diffs of different image pairs are visually comparable.
        vmax: Upper bound for the colormap. Mode-specific defaults:

            - ``"signed"``: see ``vmin``.
            - ``"abs"``: the maximum of ``|a - b|``.
            - ``"ssim"``: the maximum of the local SSIM map.

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

    luminance_a = _to_luminance(a)
    luminance_b = _to_luminance(b)

    if mode == "signed":
        signed = luminance_a - luminance_b
        given = [bound for bound in (vmin, vmax) if bound is not None]
        if len(given) == 2:
            return colorize(signed, vmin=vmin, vmax=vmax, cmap="coolwarm")
        # Zero or one bound given — produce a symmetric range. A single
        # bound's magnitude becomes the extent; otherwise it comes from
        # the data.
        extent = abs(given[0]) if given else float(np.nanmax(np.abs(signed)))
        return colorize(signed, vmin=-extent, vmax=extent, cmap="coolwarm")

    if mode == "abs":
        magnitude = np.abs(luminance_a - luminance_b)
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

        data_max = max(luminance_a.max(), luminance_b.max())
        data_min = min(luminance_a.min(), luminance_b.min())
        data_range = float(data_max - data_min) if data_max > data_min else 1.0
        _, similarity = skimage.metrics.structural_similarity(
            luminance_a, luminance_b, data_range=data_range, full=True
        )
        return colorize(similarity, vmin=vmin, vmax=vmax, cmap="viridis")

    raise ValueError(f"mode must be 'signed', 'abs' or 'ssim', but got {mode!r}")
