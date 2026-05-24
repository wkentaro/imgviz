from __future__ import annotations

from typing import Final
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._colorize import colorize


def _to_luminance(image: NDArray) -> NDArray[np.float32]:
    arr = image.astype(np.float32)
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
            ``"ssim"`` colorizes the local SSIM map (requires scikit-image and
            both image dimensions to be at least 7 pixels).
        vmin: Lower bound for the colormap. Mode-specific defaults:

            - ``"signed"``: if both ``vmin`` and ``vmax`` are ``None``, the
              bound is taken symmetrically from ``|a - b|``. If exactly one
              is given, its magnitude defines the symmetric range. If both
              are given they are used as-is (allowing asymmetric ranges).
              Zero bounds (including ``vmin=0, vmax=0``) are treated as
              missing to avoid collapsing the colormap.
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
        both_given = vmin is not None and vmax is not None
        if both_given and (vmin, vmax) != (0, 0):
            return colorize(signed, vmin=vmin, vmax=vmax, cmap="coolwarm")
        # Otherwise produce a symmetric range. A single nonzero bound's
        # magnitude becomes the extent; missing bounds and zero bounds
        # (a zero extent would collapse the colormap) fall back to the
        # data.
        nonzero = [bound for bound in (vmin, vmax) if bound not in (None, 0)]
        extent = abs(nonzero[0]) if nonzero else float(np.nanmax(np.abs(signed)))
        return colorize(signed, vmin=-extent, vmax=extent, cmap="coolwarm")

    if mode == "abs":
        magnitude = np.abs(luminance_a - luminance_b)
        if vmin is None:
            vmin = 0.0
        return colorize(magnitude, vmin=vmin, vmax=vmax, cmap="magma")

    if mode == "ssim":
        try:
            import scipy.ndimage
            import skimage.metrics
        except ImportError:
            raise ImportError(
                "scikit-image is required for mode='ssim'. "
                "Please install scikit-image or use: pip install imgviz[all]"
            ) from None

        WIN_SIZE: Final = 7
        if min(luminance_a.shape) < WIN_SIZE:
            raise ValueError(
                f"mode='ssim' requires both image dimensions to be at least "
                f"{WIN_SIZE}, but got shape {luminance_a.shape}"
            )

        nan_mask = np.isnan(luminance_a) | np.isnan(luminance_b)
        if nan_mask.all():
            return np.zeros((*luminance_a.shape, 3), dtype=np.uint8)

        # SSIM's stability constants (C1, C2) scale with data_range. When
        # both inputs share an integer dtype, use the dtype's full range so
        # scores stay comparable across different image contents; otherwise
        # (any float input or mixed dtypes) fall back to the observed
        # extent since the intrinsic range is unknown.
        if a.dtype == b.dtype and np.issubdtype(a.dtype, np.integer):
            info = np.iinfo(a.dtype)
            data_range = float(info.max - info.min)
        else:
            obs_min = min(float(np.nanmin(luminance_a)), float(np.nanmin(luminance_b)))
            obs_max = max(float(np.nanmax(luminance_a)), float(np.nanmax(luminance_b)))
            data_range = obs_max - obs_min if obs_max > obs_min else 1.0

        # scipy's uniform filter (used inside structural_similarity) uses a
        # cumsum implementation that spreads NaN across the whole image, so a
        # single NaN would collapse the entire SSIM map. Replace NaN with any
        # finite value before the call; the win_size neighborhood of every
        # original NaN is re-marked below, so the fill never reaches the
        # visible output.
        if nan_mask.any():
            sanitized_a = np.where(np.isnan(luminance_a), 0.0, luminance_a)
            sanitized_b = np.where(np.isnan(luminance_b), 0.0, luminance_b)
        else:
            sanitized_a = luminance_a
            sanitized_b = luminance_b

        _, similarity = skimage.metrics.structural_similarity(
            sanitized_a,
            sanitized_b,
            data_range=data_range,
            win_size=WIN_SIZE,
            full=True,
        )

        if nan_mask.any():
            affected = scipy.ndimage.maximum_filter(
                nan_mask.astype(np.uint8), size=WIN_SIZE
            ).astype(bool)
            similarity[affected] = np.nan

        return colorize(similarity, vmin=vmin, vmax=vmax, cmap="viridis")

    raise ValueError(f"mode must be 'signed', 'abs' or 'ssim', but got {mode!r}")
