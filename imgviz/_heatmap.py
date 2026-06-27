from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray


def heatmap(
    points: ArrayLike,
    shape: tuple[int, int],
    sigma: float = 10.0,
    weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Render a 2D Gaussian density heatmap from a list of points.

    Each point contributes an isotropic Gaussian with peak equal to its weight,
    and the contributions are summed, so denser clusters appear brighter. The
    result is a 2D float field that pipes directly into ``imgviz.colorize``.

    Args:
        points: Points (y, x) with shape (N, 2), in pixel coordinates.
        shape: Output shape (H, W).
        sigma: Standard deviation of each Gaussian, in pixels.
        weights: Optional per-point weights with shape (N,). Defaults to ones.

    Returns:
        Density field with shape (H, W); an all-zeros field when no points.

    Example:
        >>> import imgviz
        >>> density = imgviz.heatmap([(100, 150), (200, 300)], shape=(400, 600))
        >>> viz = imgviz.colorize(density)
    """
    points = np.asarray(points, dtype=float)
    if points.ndim == 1 and points.size == 0:
        points = points.reshape(0, 2)
    height, width = shape
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be a finite positive number, got {sigma}")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    if weights is None:
        weights = np.ones(len(points), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(points),):
            raise ValueError(
                f"weights must have shape ({len(points)},), got {weights.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must contain only finite values")

    # Evaluate each Gaussian only over its significant +-3 sigma window (clipped
    # to the image) so cost scales with sigma, not with the image size.
    density = np.zeros((height, width), dtype=np.float64)
    two_sigma_sq = 2.0 * sigma**2
    radius = int(np.ceil(3 * sigma))
    for (py, px), weight in zip(points, weights):
        center_y, center_x = int(np.round(py)), int(np.round(px))
        y0 = max(0, center_y - radius)
        y1 = min(height, center_y + radius + 1)
        x0 = max(0, center_x - radius)
        x1 = min(width, center_x + radius + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        ys = np.arange(y0, y1)[:, np.newaxis]
        xs = np.arange(x0, x1)[np.newaxis, :]
        window = np.exp(-((ys - py) ** 2 + (xs - px) ** 2) / two_sigma_sq)
        density[y0:y1, x0:x1] += weight * window
    return density
