from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from . import color as color_module
from . import draw as draw_module
from . import label as label_module
from . import utils


def mask_to_bbox(masks: NDArray | Sequence[NDArray]) -> NDArray[np.floating]:
    warnings.warn(
        "'mask_to_bbox' is deprecated. Use 'masks_to_bboxes'",
        DeprecationWarning,
    )
    bboxes = masks_to_bboxes(masks)
    bboxes[:, 2:] += 1  # tight to loose
    return bboxes


def masks_to_bboxes(masks: NDArray | Sequence[NDArray]) -> NDArray[np.floating]:
    """Convert mask to tight bounding box.

    Parameters
    ----------
    masks
        Boolean masks with shape (N, H, W).

    Returns
    -------
    bboxes
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
    labels: Sequence[int],
    bboxes: NDArray | None = None,
    masks: NDArray[np.bool_] | Sequence[NDArray[np.bool_]] | None = None,
    captions: Sequence[str | None] | None = None,
    font_size: int = 25,
    line_width: int = 5,
    boundary_width: int = 1,
    alpha: float = 0.7,
    colormap: NDArray[np.uint8] | None = None,
    font_path: str | None = None,
) -> NDArray[np.uint8]:
    """Convert instances to rgb.

    Parameters
    ----------
    image
        RGB image with shape (H, W, 3).
    labels
        Labels with length N.
    bboxes
        Bounding boxes with shape (N, 4).
    masks
        Masks with shape (N, H, W).
    captions
        Captions with length N.
    font_size
        Font size.
    line_width
        Line width.
    boundary_width
        Boundary width.
    alpha
        Alpha of RGB.
    colormap
        Label id to RGB color.
    font_path
        Font path.

    Returns
    -------
    dst
        Visualized image with shape (H, W, 3).

    """
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8

    if image.ndim == 2:
        image = color_module.gray2rgb(image)
    assert image.ndim == 3

    assert all(label_i >= 0 for label_i in labels)

    n_instance = len(labels)

    if bboxes is None:
        assert masks is not None
        bboxes = masks_to_bboxes(masks=masks)
    if captions is None:
        captions = [None] * n_instance

    assert len(bboxes) == len(captions) == n_instance

    if colormap is None:
        colormap = label_module.label_colormap()

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

        try:
            import skimage.segmentation

            boundary = skimage.segmentation.find_boundaries(mask, connectivity=2)
            for _ in range(boundary_width - 1):
                boundary = skimage.morphology.binary_dilation(boundary)
            dst[boundary] = (200, 200, 200)
        except ImportError:
            pass

    dst = utils.numpy_to_pillow(dst)
    for instance_id in range(n_instance):
        bbox = bboxes[instance_id]
        label = labels[instance_id]
        caption = captions[instance_id]

        color_cls = colormap[label % len(colormap)]

        y1, x1, y2, x2 = bbox
        if (y2 - y1) * (x2 - x1) == 0:
            continue

        aabb1 = np.array([y1, x1], dtype=int)
        aabb2 = np.array([y2, x2], dtype=int)
        draw_module.rectangle_(
            dst,
            aabb1,
            aabb2,
            outline=color_cls,
            width=line_width,
        )

        if caption is not None:
            for loc in ["lt+", "lt"]:
                y1, x1, y2, x2 = draw_module.text_in_rectangle_aabb(
                    img_shape=(dst.height, dst.width),
                    loc=loc,
                    text=caption,
                    size=font_size,
                    aabb1=aabb1,
                    aabb2=aabb2,
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
                aabb1=aabb1,
                aabb2=aabb2,
                font_path=font_path,
            )
    return utils.pillow_to_numpy(dst)
