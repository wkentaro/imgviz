from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray


def blend(
    under: NDArray[np.uint8], over: ArrayLike, alpha: float | NDArray
) -> NDArray[np.uint8]:
    """Alpha-composite ``over`` onto ``under`` and return a new uint8 array.

    ``over`` is an image or a color that broadcasts against ``under``; ``alpha``
    is a scalar or an array that broadcasts the same way.
    """
    alpha_arr = np.asarray(alpha, dtype=float)
    if not ((0 <= alpha_arr) & (alpha_arr <= 1)).all():
        raise ValueError(f"alpha must be in range [0, 1], got {alpha}")
    mixed = (1 - alpha_arr) * under.astype(float) + alpha_arr * np.asarray(
        over, dtype=float
    )
    # Round before the cast: a plain uint8 cast truncates and biases every
    # .5 result down by one.
    return np.clip(mixed.round(), 0, 255).astype(np.uint8)
