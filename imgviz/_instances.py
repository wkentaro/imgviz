from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from . import _color
from . import _label
from . import _utils
from . import draw as draw_module


def masks_to_bboxes(masks: NDArray | Sequence[NDArray]) -> NDArray[np.floating]:
    """Convert mask to tight bounding box.

    Args:
        masks: Boolean masks with shape (N, H, W).

    Returns:
        Tight bounding boxes with shape (N, 4). [(ymin, xmin, ymax, xmax), ...]
        where both left-top and right-bottom are inclusive.
    """
    bboxes = np.zeros((len(masks), 4), dtype=float)
    for i, mask in enumerate(masks):
        if mask.sum() == 0:
            continue
        where = np.argwhere(mask)
        (ymin, xmin), (ymax, xmax) = where.min(axis=0), where.max(axis=0)
        bbox = ymin, xmin, ymax, xmax
        bboxes[i] = bbox
    return bboxes


def instances2rgb(
    image: NDArray[np.uint8],
    labels: Sequence[int] | NDArray[np.integer],
    bboxes: NDArray | None = None,
    masks: NDArray[np.bool_] | Sequence[NDArray[np.bool_]] | None = None,
    captions: Sequence[str | None] | None = None,
    font_size: int = 25,
    line_width: int = 5,
    boundary_width: int = 0,
    alpha: float = 0.5,
    colormap: NDArray[np.uint8] | None = None,
    font_path: str | None = None,
) -> NDArray[np.uint8]:
    """Convert instances to rgb.

    Args:
        image: RGB image with shape (H, W, 3).
        labels: Labels with length N.
        bboxes: Bounding boxes with shape (N, 4).
        masks: Masks with shape (N, H, W).
        captions: Captions with length N.
        font_size: Font size.
        line_width: Line width.
        boundary_width: Boundary width.
        alpha: Alpha of RGB.
        colormap: Label id to RGB color.
        font_path: Font path.

    Returns:
        Visualized image with shape (H, W, 3).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, but got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"image dtype must be np.uint8, but got {image.dtype}")

    if image.ndim == 2:
        image = _color.gray2rgb(image)
    if image.ndim != 3:
        raise ValueError(f"image must be 2 or 3 dimensional, but got {image.ndim}")

    negative_labels = [label_i for label_i in labels if label_i < 0]
    if negative_labels:
        raise ValueError(f"all labels must be >= 0, but got {negative_labels}")

    n_instance = len(labels)

    if bboxes is None:
        if masks is None:
            raise ValueError("bboxes or masks must be provided")
        bboxes = masks_to_bboxes(masks=masks)
    if captions is None:
        captions = [None] * n_instance

    if len(bboxes) != n_instance or len(captions) != n_instance:
        raise ValueError(
            f"bboxes, captions and labels must have the same length, "
            f"but got {len(bboxes)=}, {len(captions)=}, {n_instance=}"
        )

    if colormap is None:
        colormap = _label.label_colormap()

    dst = image

    for instance_id in range(n_instance):
        if masks is None:
            continue

        mask: NDArray[np.bool_] = masks[instance_id]
        if mask.sum() == 0:
            continue

        color_ins = colormap[1:][instance_id % len(colormap[1:])]

        maskviz = mask[:, :, None] * color_ins.astype(float)
        dst = dst.copy()
        dst[mask] = (1 - alpha) * image[mask].astype(float) + alpha * maskviz[mask]

        if boundary_width > 0:
            try:
                import skimage.segmentation
            except ImportError:
                raise ImportError(
                    "skimage is required for boundary_width > 0. "
                    "Please install scikit-image or use: pip install imgviz[all]"
                ) from None

            boundary = skimage.segmentation.find_boundaries(mask, connectivity=2)
            for _ in range(boundary_width - 1):
                boundary = skimage.morphology.binary_dilation(boundary)
            dst[boundary] = (200, 200, 200)

    dst = _utils.numpy_to_pillow(dst)
    for instance_id in range(n_instance):
        bbox = bboxes[instance_id]
        label = labels[instance_id]
        caption = captions[instance_id]

        color_cls = colormap[label % len(colormap)]

        y1, x1, y2, x2 = bbox
        if (y2 - y1) * (x2 - x1) == 0:
            continue

        yx1 = np.array([y1, x1], dtype=int)
        yx2 = np.array([y2, x2], dtype=int)
        draw_module.rectangle_(
            dst,
            yx1,
            yx2,
            outline=color_cls,
            width=line_width,
        )

        if caption is not None:
            locs: tuple[Literal["lt+"], Literal["lt"]] = ("lt+", "lt")
            for loc in locs:
                y1, x1, y2, x2 = draw_module.text_in_rectangle_aabb(
                    img_shape=(dst.height, dst.width),
                    loc=loc,
                    text=caption,
                    size=font_size,
                    yx1=yx1,
                    yx2=yx2,
                    font_path=font_path,
                )
                if y1 >= 0 and x1 >= 0 and y2 < dst.height and x2 < dst.width:
                    break
            draw_module.text_in_rectangle_(
                img=dst,
                loc=loc,
                text=caption,
                size=font_size,
                background=color_cls,
                yx1=yx1,
                yx2=yx2,
                font_path=font_path,
            )
    return _utils.pillow_to_numpy(dst)
