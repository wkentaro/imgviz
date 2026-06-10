from __future__ import annotations

import math
from typing import Final
from typing import Literal

import cmap as _cmap
import numpy as np
from numpy.typing import NDArray

from . import draw as draw_module
from ._color import get_fg_color


def _pick_nice_length(target_units: float) -> float:
    """Largest of {1, 2, 5} times a power of 10 not exceeding target_units."""
    exponent = math.floor(math.log10(target_units))
    fraction = target_units / 10**exponent
    nice = 5 if fraction >= 5 else 2 if fraction >= 2 else 1
    return nice * 10**exponent


def scalebar(
    image: NDArray[np.uint8],
    pixels_per_unit: float,
    unit: str = "m",
    loc: Literal["lt", "rt", "lb", "rb"] = "rb",
    color: Literal["auto"] | _cmap.ColorLike = "auto",
    font_path: str | None = None,
) -> NDArray[np.uint8]:
    """Draw a scale bar with a "nice" length and unit label into an image.

    Given a pixel-to-physical conversion, picks a length of 1, 2, or 5 times a
    power of 10 that fits within ~20% of the image width, then draws a
    horizontal bar with a label such as "5 m" or "200 nm".

    Args:
        image: Input uint8 RGB image with shape (H, W, 3).
        pixels_per_unit: Number of pixels that span one physical unit.
        unit: Physical unit shown in the label (e.g. "m", "nm").
        loc: Corner to place the scale bar: "lt", "rt", "lb", or "rb".
        color: Bar and label color. "auto" picks black or white from the
            luminance of the region the bar and label cover; otherwise any
            cmap-compatible color.
        font_path: Optional path to a font file for the label.

    Returns:
        Image with the scale bar drawn, same shape and dtype as the input.

    Example:
        >>> import imgviz
        >>> image = imgviz.data.arc2017()["rgb"]
        >>> viz = imgviz.scalebar(image, pixels_per_unit=200, unit="m")
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must be uint8 RGB (H, W, 3), got {image.shape} {image.dtype}"
        )
    if pixels_per_unit <= 0 or not math.isfinite(pixels_per_unit):
        raise ValueError(
            f"pixels_per_unit must be a finite positive number, got {pixels_per_unit}"
        )
    if loc not in ("lt", "rt", "lb", "rb"):
        raise ValueError(f"loc must be one of lt, rt, lb, rb, got {loc!r}")

    BAR_BUDGET: Final = 0.2
    MARGIN_RATIO: Final = 0.04
    FONT_RATIO: Final = 0.03
    BAR_THICKNESS_RATIO: Final = 0.008
    GAP_RATIO: Final = 0.006

    height, width = image.shape[:2]
    margin = round(MARGIN_RATIO * width)
    font_size = max(12, round(FONT_RATIO * height))
    bar_thickness = max(3, round(BAR_THICKNESS_RATIO * height))
    gap = max(2, round(GAP_RATIO * height))

    target_units = BAR_BUDGET * width / pixels_per_unit
    nice_units = _pick_nice_length(target_units)
    bar_length = round(nice_units * pixels_per_unit)

    label = f"{nice_units:g} {unit}"
    label_height, label_width = draw_module.text_size(
        text=label, size=font_size, font_path=font_path
    )

    block_width = max(bar_length, label_width)
    block_height = label_height + gap + bar_thickness
    x0 = width - margin - block_width if loc in ("rt", "rb") else margin
    y0 = height - margin - block_height if loc in ("lb", "rb") else margin

    if isinstance(color, str) and color == "auto":
        region = image[max(0, y0) : y0 + block_height, max(0, x0) : x0 + block_width]
        if region.size == 0:
            region = image
        rgb = get_fg_color(region.mean(axis=(0, 1)).astype(np.uint8))
    else:
        rgb = tuple(_cmap.Color(color).rgba8[:3])

    label_x = x0 + (block_width - label_width) / 2
    bar_x = x0 + (block_width - bar_length) / 2
    bar_y = y0 + label_height + gap

    viz = draw_module.text(
        image,
        yx=(y0, label_x),
        text=label,
        size=font_size,
        color=rgb,
        font_path=font_path,
    )
    viz = draw_module.rectangle(
        viz,
        yx1=(bar_y, bar_x),
        yx2=(bar_y + bar_thickness, bar_x + bar_length),
        fill=rgb,
    )
    return viz
