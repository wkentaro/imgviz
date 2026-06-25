from __future__ import annotations

from typing import Final
from typing import Literal
from typing import TypeAlias
from typing import TypeVar
from typing import overload

import numpy as np
from numpy.typing import NDArray

Kind: TypeAlias = Literal["protanopia", "deuteranopia", "tritanopia"]

_FloatT = TypeVar("_FloatT", bound=np.floating)


@overload
def colorblind(image: NDArray[np.uint8], kind: Kind = ...) -> NDArray[np.uint8]: ...


@overload
def colorblind(image: NDArray[_FloatT], kind: Kind = ...) -> NDArray[_FloatT]: ...


def colorblind(image: NDArray, kind: Kind = "deuteranopia") -> NDArray:
    """Simulate how an image looks to a viewer with color-vision deficiency.

    Applies a single dichromacy simulation matrix (Vienot et al. 1999, as
    popularized by Wickline's daltonize) directly to the sRGB values.

    Args:
        image: Input RGB image with shape (H, W, 3), either uint8 in [0, 255]
            or float in [0, 1].
        kind: Form of color-vision deficiency to simulate, one of
            "protanopia", "deuteranopia", or "tritanopia".

    Returns:
        Simulated image with the same shape and dtype as the input.

    Example:
        >>> import imgviz
        >>> image = imgviz.data.arc2017()["rgb"]
        >>> simulated = imgviz.colorblind(image, kind="deuteranopia")
    """
    # Vienot et al. 1999 dichromacy matrices applied directly to sRGB (gamma
    # linearization intentionally omitted for a single matrix multiply).
    MATRICES: Final[dict[str, list[list[float]]]] = {
        "protanopia": [
            [0.56667, 0.43333, 0.0],
            [0.55833, 0.44167, 0.0],
            [0.0, 0.24167, 0.75833],
        ],
        "deuteranopia": [
            [0.625, 0.375, 0.0],
            [0.70, 0.30, 0.0],
            [0.0, 0.30, 0.70],
        ],
        "tritanopia": [
            [0.95, 0.05, 0.0],
            [0.0, 0.43333, 0.56667],
            [0.0, 0.475, 0.525],
        ],
    }

    if kind not in MATRICES:
        raise ValueError(f"kind must be one of {sorted(MATRICES)}, got {kind!r}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be RGB with shape (H, W, 3), got {image.shape}")
    is_float = np.issubdtype(image.dtype, np.floating)
    if image.dtype != np.uint8 and not is_float:
        raise ValueError(f"image must be uint8 or float, got {image.dtype}")

    matrix = np.asarray(MATRICES[kind])
    simulated = image.astype(np.float64) @ matrix.T

    # Rows sum to 1 with non-negative entries, so in-range inputs stay in range;
    # the clip guards float rounding and any out-of-[0, 1] float input.
    max_value = 1.0 if is_float else 255.0
    simulated = simulated.clip(0, max_value)
    if is_float:
        return simulated.astype(image.dtype)
    return simulated.round().astype(np.uint8)
