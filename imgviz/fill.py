from __future__ import annotations

import abc
import dataclasses

import numpy as np
from numpy.typing import NDArray


@dataclasses.dataclass
class Fill(abc.ABC):
    """Abstract base class for mask fill specifications.

    Subclasses must implement __call__ to render the fill onto an image.

    """

    @abc.abstractmethod
    def __call__(
        self,
        mask: NDArray[np.bool_],
        image: NDArray[np.uint8],
        alpha: float = 1.0,
        copy: bool = True,
    ) -> NDArray[np.uint8]:
        """Apply the fill to masked region of an image.

        Args:
            mask: Boolean mask (H, W) specifying region to fill.
            image: Background image (H, W, 3) to blend onto.
            alpha: Opacity of fill (0.0 to 1.0). Default 1.0 (fully opaque).
            copy: If True, make a copy of the image before applying the fill.

        Returns:
            Image with filled mask region (H, W, 3).
        """
        pass


@dataclasses.dataclass
class Solid(Fill):
    """Solid color fill.

    Args:
        color: RGB color as (R, G, B) tuple or numpy array.

    Examples:
        >>> fill = imgviz.fill.Solid(color=(255, 0, 0))
        >>> result = imgviz.mask2rgb(mask, image=rgb, fill=fill)
    """

    color: tuple[int, int, int] | NDArray[np.uint8]

    def __call__(
        self,
        mask: NDArray[np.bool_],
        image: NDArray[np.uint8],
        alpha: float = 1.0,
        copy: bool = True,
    ) -> NDArray[np.uint8]:
        return _blend(image=image, mask=mask, color=self.color, alpha=alpha, copy=copy)


@dataclasses.dataclass
class Stripe(Fill):
    """Stripe pattern fill.

    Fills the masked region with diagonal, horizontal, or vertical stripes.

    Args:
        color: RGB color as (R, G, B) tuple or numpy array.
        angle: Stripe angle in radians. 0 = horizontal, pi/4 = diagonal
            (top-right to bottom-left), pi/2 = vertical. Default pi/4.
        width: Width of each stripe in pixels. Must be positive. Default 3.
        gap: Gap between stripes in pixels. Must be non-negative. Default 9.

    Examples:
        >>> fill = imgviz.fill.Stripe(color=(255, 0, 0), angle=0, width=2, gap=4)
        >>> result = imgviz.mask2rgb(mask, image=rgb, fill=fill)
    """

    color: tuple[int, int, int] | NDArray[np.uint8]
    angle: float = np.deg2rad(45)  # [rad]
    width: int = 3  # [px]
    gap: int = 9  # [px]

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.gap < 0:
            raise ValueError(f"gap must be non-negative, got {self.gap}")

    def __call__(
        self,
        mask: NDArray[np.bool_],
        image: NDArray[np.uint8],
        alpha: float = 1.0,
        copy: bool = True,
    ) -> NDArray[np.uint8]:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return image.copy() if copy else image

        distance: NDArray[np.float64] = xs.astype(np.float64) * np.sin(
            self.angle
        ) + ys.astype(np.float64) * np.cos(self.angle)

        period: int = self.width + self.gap
        stripe_on: NDArray[np.bool_] = (distance % period) < self.width

        stripe_mask: NDArray[np.bool_] = np.zeros_like(mask, dtype=bool)
        stripe_mask[ys[stripe_on], xs[stripe_on]] = True
        return _blend(
            image=image,
            mask=mask & stripe_mask,
            color=self.color,
            alpha=alpha,
            copy=copy,
        )


def _blend(
    image: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    color: tuple[int, int, int] | NDArray[np.uint8],
    alpha: float,
    copy: bool,
) -> NDArray[np.uint8]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in range [0.0, 1.0], got {alpha}")
    if copy:
        image = image.copy()
    image[mask] = np.clip(
        (
            (1 - alpha) * image[mask].astype(np.float32)
            + alpha * np.array(color, dtype=np.float32)
        ).round(),
        0,
        255,
    ).astype(np.uint8)
    return image
