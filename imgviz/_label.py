from __future__ import annotations

import functools
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from . import _color
from . import _utils
from . import draw as draw_module


def _bitget(byte_value: NDArray[np.uint8], idx: int) -> NDArray[np.uint8]:
    shape: tuple[int, ...] = byte_value.shape + (8,)
    return np.unpackbits(byte_value).reshape(shape)[..., -1 - idx]


@functools.lru_cache(maxsize=1)
def label_colormap() -> NDArray[np.uint8]:
    """Label colormap for maximum 256 labels.

    Returns:
        Label id to colormap with shape (256, 3).
    """
    i = np.arange(256, dtype=np.uint8)
    i = np.repeat(i[:, None], 8, axis=1)
    i = np.right_shift(i, np.arange(0, 24, 3)).astype(np.uint8)
    j = np.arange(8)[::-1]
    r = np.bitwise_or.reduce(np.left_shift(_bitget(i, 0), j), axis=1)
    g = np.bitwise_or.reduce(np.left_shift(_bitget(i, 1), j), axis=1)
    b = np.bitwise_or.reduce(np.left_shift(_bitget(i, 2), j), axis=1)

    cmap: NDArray[np.uint8] = np.stack((r, g, b), axis=1).astype(np.uint8)
    cmap.setflags(write=False)
    return cmap


def label2rgb(
    label: NDArray,
    image: NDArray[np.uint8] | None = None,
    alpha: float | list[float] | dict[int, float] = 0.5,
    label_names: list[str] | dict[int, str] | None = None,
    font_size: int = 30,
    thresh_suppress: float = 0,
    colormap: NDArray[np.uint8] | None = None,
    loc: Literal["centroid", "lt", "rt", "lb", "rb"] = "rb",
    font_path: str | None = None,
) -> NDArray[np.uint8]:
    """Convert label to rgb.

    Args:
        label: Label image with shape (H, W).
        image: RGB image with shape (H, W, 3).
        alpha: Alpha of RGB. If given as a list or dict, it is treated as alpha
            for each class according to the index or key.
        label_names: Label id to label name.
        font_size: Font size.
        thresh_suppress: Threshold of label ratio in the label image.
        colormap: Label id to color. By default, :func:`~imgviz.label_colormap`
            is used.
        loc: Location of legend ('centroid', 'lt', 'rt', 'lb', 'rb').
        font_path: Font path.

    Returns:
        Visualized image with shape (H, W, 3).
    """
    if colormap is None:
        colormap = label_colormap()

    if label.dtype == bool:
        label = label.astype(np.int32)

    res = colormap[label]

    random_state = np.random.RandomState(seed=1234)

    mask_unlabeled = label < 0
    res[mask_unlabeled] = random_state.rand(*(mask_unlabeled.sum(), 3)) * 255

    unique_labels = np.unique(label)
    max_label_id = unique_labels[-1]

    if isinstance(alpha, (int, float)):
        alpha_arr = np.array([alpha for _ in range(max_label_id + 1)])
    elif isinstance(alpha, dict):
        alpha_arr = np.array(
            [alpha.get(label_id, 0.5) for label_id in range(max_label_id + 1)]
        )
    else:
        alpha_arr = np.asarray(alpha)
        if alpha_arr.ndim != 1:
            raise ValueError(f"alpha must be 1D array, but got {alpha_arr.ndim}D")
    if not ((0 <= alpha_arr) & (alpha_arr <= 1)).all():
        raise ValueError("alpha values must be in [0, 1]")
    alpha_map = alpha_arr[label][:, :, None]

    if image is not None:
        if image.ndim == 2:
            image = _color.gray2rgb(image)
        res = (1 - alpha_map) * image.astype(float) + alpha_map * res.astype(float)
        res = np.clip(res.round(), 0, 255).astype(np.uint8)

    if label_names is None:
        return res

    unique_labels = unique_labels[unique_labels != -1]
    if isinstance(label_names, dict):
        unique_labels = [
            label_id for label_id in unique_labels if label_names.get(label_id)
        ]
    else:
        unique_labels = [
            label_id for label_id in unique_labels if label_names[label_id]
        ]
    if len(unique_labels) == 0:
        return res

    if loc == "centroid":
        random_state: np.random.RandomState = np.random.RandomState(0)

        res = _utils.numpy_to_pillow(res)
        for label_i in unique_labels:
            mask = label == label_i
            if 1.0 * mask.sum() / mask.size < thresh_suppress:
                continue
            y: int
            x: int
            y, x = np.array(_center_of_mass(mask)).round().astype(int).tolist()

            if label[y, x] != label_i:
                Y, X = np.where(mask)
                point_index = random_state.randint(0, len(Y))
                y, x = Y[point_index].item(), X[point_index].item()

            text = label_names[label_i]
            height, width = draw_module.text_size(
                text, size=font_size, font_path=font_path
            )
            color = _color.get_fg_color(res.getpixel((x, y)))
            draw_module.text_(
                res,
                yx=(y - height // 2, x - width // 2),
                text=text,
                color=color,
                size=font_size,
                font_path=font_path,
            )
    elif loc in ["rb", "lt", "rt", "lb"]:
        text_sizes = np.array(
            [
                draw_module.text_size(
                    label_names[label_id], font_size, font_path=font_path
                )
                for label_id in unique_labels
            ]
        )
        text_height, text_width = text_sizes.max(axis=0)
        pad: int = max(2, font_size // 6)
        legend_height = text_height * len(unique_labels) + pad
        legend_width = text_width + text_height + 2 * pad

        height, width = label.shape[:2]
        if loc == "rb":
            aabb2 = np.array([height - pad, width - pad], dtype=float)
            aabb1 = aabb2 - (legend_height, legend_width)
        elif loc == "lt":
            aabb1 = np.array([pad, pad], dtype=float)
            aabb2 = aabb1 + (legend_height, legend_width)
        elif loc == "rt":
            aabb1 = np.array([pad, width - pad - legend_width], dtype=float)
            aabb2 = aabb1 + (legend_height, legend_width)
        elif loc == "lb":
            aabb2 = np.array([height - pad, pad + legend_width], dtype=float)
            aabb1 = aabb2 - (legend_height, legend_width)
        else:
            raise ValueError(f"unsupported loc: {loc}")

        alpha = 0.5
        y1, x1 = aabb1.round().astype(int)
        y2, x2 = aabb2.round().astype(int)
        res[y1:y2, x1:x2] = alpha * res[y1:y2, x1:x2] + alpha * 255

        box_size = text_height - 2 * pad
        res = _utils.numpy_to_pillow(res)
        for i, label_id in enumerate(unique_labels):
            box_aabb1 = aabb1 + (i * text_height + pad, pad)
            box_aabb2 = box_aabb1 + (box_size, box_size)
            draw_module.rectangle_(
                res, aabb1=box_aabb1, aabb2=box_aabb2, fill=colormap[label_id]
            )
            draw_module.text_(
                res,
                yx=aabb1 + (i * text_height, text_height),
                text=label_names[label_id],
                size=font_size,
                font_path=font_path,
            )
    else:
        raise ValueError(f"unsupported loc: {loc}")

    return _utils.pillow_to_numpy(res)


def _center_of_mass(mask: NDArray[np.bool_]) -> tuple[float, float]:
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError(
            f"mask must be 2D bool array, but got {mask.ndim}D {mask.dtype}"
        )
    mask_float: NDArray[np.float32] = mask.astype(np.float32) / mask.sum()
    dx = np.sum(mask_float, 0)
    dy = np.sum(mask_float, 1)
    cx = np.sum(dx * np.arange(mask.shape[1]))
    cy = np.sum(dy * np.arange(mask.shape[0]))
    return cy.item(), cx.item()
