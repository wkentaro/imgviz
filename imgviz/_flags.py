from __future__ import annotations

from collections.abc import Sequence
from typing import Final
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from . import _color
from . import _label
from . import _utils
from . import components
from . import draw as draw_module


def flags2rgb(
    image: NDArray[np.uint8],
    flags: NDArray[np.bool_],
    centers: NDArray[np.floating],
    flag_names: Sequence[str],
    diameter: float = 30,
    flag_colors: NDArray[np.uint8] | None = None,
    wedges: Literal["on", "all"] = "on",
    font_size: int = 25,
    font_path: str | None = None,
    loc: Literal["lt", "rt", "lb", "rb"] = "rb",
    outline: draw_module.Ink | None = (255, 255, 255),
    outline_width: int = 1,
) -> NDArray[np.uint8]:
    """Visualize per-instance boolean flags as pie glyphs with a legend.

    Wedge angle encodes flag identity, never quantity. With ``wedges="on"``,
    color identifies the flag and angle is just packing: only active flags get
    wedges, packed clockwise in flag-index order (zero active flags draws no
    pie, one draws a solid disc). With ``wedges="all"``, both color and angle
    identify the flag: every flag keeps a fixed wedge at a fixed angle, drawn
    light gray when off, and the legend gains one synthetic ("off", gray)
    entry. Around 5 wedges stay legible at the default diameter.

    Args:
        image: Image with shape (H, W) or (H, W, 3).
        flags: Boolean flags with shape (N, F).
        centers: Pie centers with shape (N, 2). [(cy, cx), ...]
        flag_names: Flag names with length F.
        diameter: Diameter of each pie in pixels.
        flag_colors: Color for each flag with shape (F, 3). By default,
            :func:`~imgviz.label_colormap` rows 1 to F are used.
        wedges: Which flags get wedges ('on', 'all').
        font_size: Font size of the legend.
        font_path: Font path.
        loc: Location of legend ('lt', 'rt', 'lb', 'rb').
        outline: Color for the pie and wedge edges. None for no outline.
        outline_width: Width of the pie outline in pixels.

    Returns:
        Visualized image with shape (H, W, 3).
    """
    OFF_COLOR: Final = (200, 200, 200)

    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, but got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"image dtype must be np.uint8, but got {image.dtype}")
    if image.ndim == 2:
        image = _color.gray2rgb(image)
    if image.ndim != 3:
        raise ValueError(f"image must be 2 or 3 dimensional, but got {image.ndim}")

    if not isinstance(flags, np.ndarray):
        raise TypeError(f"flags must be a numpy array, but got {type(flags).__name__}")
    if flags.dtype != bool:
        raise ValueError(f"flags dtype must be bool, but got {flags.dtype}")
    if flags.ndim != 2:
        raise ValueError(f"flags must be 2 dimensional (N, F), but got {flags.ndim}")
    n_instances, n_flags = flags.shape

    centers = np.asarray(centers)
    if centers.shape != (n_instances, 2):
        raise ValueError(
            f"centers shape must be (N, 2) matching {n_instances} instances, "
            f"but got {centers.shape}"
        )

    if len(flag_names) != n_flags:
        raise ValueError(
            f"flag_names must have one name per flag, "
            f"but got {len(flag_names)=}, {n_flags=}"
        )

    if flag_colors is None:
        flag_colors = _label.label_colormap()[1 : n_flags + 1]
    if flag_colors.shape != (n_flags, 3):
        raise ValueError(
            f"flag_colors shape must be ({n_flags}, 3), but got {flag_colors.shape}"
        )

    if wedges not in ("on", "all"):
        raise ValueError(f"unsupported wedges: {wedges}")

    dst = _utils.numpy_to_pillow(image)
    for i in range(n_instances):
        fills: list[draw_module.Ink]
        if wedges == "on":
            fills = [flag_colors[j] for j in range(n_flags) if flags[i, j]]
        else:
            fills = [
                flag_colors[j] if flags[i, j] else OFF_COLOR for j in range(n_flags)
            ]
        if not fills:
            continue
        cy, cx = centers[i]
        draw_module.pie_(
            image=dst,
            center=(cy, cx),
            diameter=diameter,
            fills=fills,
            outline=outline,
            width=outline_width,
        )

    items: list[components.LegendItem] = list(zip(flag_names, flag_colors))
    if wedges == "all":
        items.append(("off", OFF_COLOR))
    components.legend_(
        image=dst, items=items, font_size=font_size, font_path=font_path, loc=loc
    )
    return _utils.pillow_to_numpy(dst)
