from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
from typing import TypeAlias

import numpy as np
import PIL.Image
from numpy.typing import NDArray

from .. import _utils
from .. import draw as draw_module
from ..draw import Ink

LegendItem: TypeAlias = tuple[str, Ink]


def legend(
    image: NDArray[np.uint8],
    items: Sequence[LegendItem],
    font_size: int = 25,
    font_path: str | None = None,
    loc: Literal["lt", "rt", "lb", "rb"] = "rb",
) -> NDArray[np.uint8]:
    """Draw a corner legend of colored boxes with text labels.

    Args:
        image: Input image with shape (H, W, 3).
        items: Resolved (text, color) pairs, one per legend row. If empty, the
            image is returned unchanged.
        font_size: Font size.
        font_path: Font path.
        loc: Corner to place the legend ('lt', 'rt', 'lb', 'rb').

    Returns:
        Output image.
    """
    dst = _utils.numpy_to_pillow(image)
    legend_(
        image=dst,
        items=items,
        font_size=font_size,
        font_path=font_path,
        loc=loc,
    )
    return _utils.pillow_to_numpy(dst)


def legend_(
    image: PIL.Image.Image,
    items: Sequence[LegendItem],
    font_size: int = 25,
    font_path: str | None = None,
    loc: Literal["lt", "rt", "lb", "rb"] = "rb",
) -> None:
    """Draw a corner legend on a PIL image in-place.

    Args:
        image: PIL image to draw on (modified in-place).
        items: Resolved (text, color) pairs, one per legend row. If empty, this
            is a no-op.
        font_size: Font size.
        font_path: Font path.
        loc: Corner to place the legend ('lt', 'rt', 'lb', 'rb').
    """
    if len(items) == 0:
        return

    text_sizes = np.array(
        [
            draw_module.text_size(text, font_size, font_path=font_path)
            for text, _ in items
        ]
    )
    text_height, text_width = text_sizes.max(axis=0)
    pad: int = max(2, font_size // 6)
    legend_height = text_height * len(items) + pad
    legend_width = text_width + text_height + 2 * pad

    width, height = image.size
    if loc == "rb":
        yx2 = np.array([height - pad, width - pad], dtype=float)
        yx1 = yx2 - (legend_height, legend_width)
    elif loc == "lt":
        yx1 = np.array([pad, pad], dtype=float)
        yx2 = yx1 + (legend_height, legend_width)
    elif loc == "rt":
        yx1 = np.array([pad, width - pad - legend_width], dtype=float)
        yx2 = yx1 + (legend_height, legend_width)
    elif loc == "lb":
        yx2 = np.array([height - pad, pad + legend_width], dtype=float)
        yx1 = yx2 - (legend_height, legend_width)
    else:
        raise ValueError(f"unsupported loc: {loc}")

    alpha = 0.5
    y1, x1 = yx1.round().astype(int)
    y2, x2 = yx2.round().astype(int)
    region = np.asarray(image)[y1:y2, x1:x2]
    washed = (alpha * region + alpha * 255).round().astype(np.uint8)
    image.paste(_utils.numpy_to_pillow(washed), (int(x1), int(y1)))

    box_size = text_height - 2 * pad
    for i, (text, color) in enumerate(items):
        box_yx1 = yx1 + (i * text_height + pad, pad)
        box_yx2 = box_yx1 + (box_size, box_size)
        draw_module.rectangle_(image, yx1=box_yx1, yx2=box_yx2, fill=color)
        draw_module.text_(
            image,
            yx=yx1 + (i * text_height, text_height),
            text=text,
            size=font_size,
            font_path=font_path,
        )
