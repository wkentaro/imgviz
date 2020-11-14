import numpy as np

from . import draw as draw_module
from . import label as label_module


def mask_to_bbox(masks):
    bboxes = np.zeros((len(masks), 4), dtype=float)
    for i, mask in enumerate(masks):
        if mask.sum() == 0:
            continue
        where = np.argwhere(mask)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        bbox = y1, x1, y2, x2
        bboxes[i] = bbox
    return bboxes


def instances2rgb(
    image,
    labels,
    bboxes=None,
    masks=None,
    captions=None,
    font_size=25,
    line_width=5,
    boundary_width=1,
    alpha=0.7,
    colormap=None,
    font_path=None,
):
    """Convert instances to rgb.

    Parameters
    ----------
    image: numpy.ndarray, (H, W, 3), numpy.uint8
        RGB image.
    labels: list of int, (N,)
        Labels.
    bboxes: list of numpy.ndarray, (N, 4), float
        Bounding boxes.
    masks: numpy.ndarray, (N, H, W), bool
        Masks.
    captions: list of str
        Captions.
    font_size: int
        Font size.
    line_width: int
        Line width.
    alpha: float
        Alpha of RGB.
    colormap: numpy.ndarray, (M, 3), numpy.uint8
        Label id to RGB color.

    Returns
    -------
    dst: numpy.ndarray, (H, W, 3), numpy.uint8
        Visualized image.

    """
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8
    assert image.ndim == 3

    assert all(label_i >= 0 for label_i in labels)

    n_instance = len(labels)

    if masks is None:
        assert bboxes is not None
        masks = [None] * n_instance
    if bboxes is None:
        assert masks is not None
        bboxes = mask_to_bbox(masks)
    if captions is None:
        captions = [None] * n_instance

    assert len(masks) == len(bboxes) == len(captions) == n_instance

    if colormap is None:
        colormap = label_module.label_colormap()

    dst = image

    for instance_id in range(n_instance):
        mask = masks[instance_id]

        if mask is None or mask.sum() == 0:
            continue

        color_ins = colormap[1:][instance_id % len(colormap[1:])]

        maskviz = mask[:, :, None] * color_ins.astype(float)
        dst = dst.copy()
        dst[mask] = (1 - alpha) * image[mask].astype(float) + alpha * maskviz[
            mask
        ]

        try:
            import skimage.segmentation

            boundary = skimage.segmentation.find_boundaries(
                mask, connectivity=2
            )
            for _ in range(boundary_width - 1):
                boundary = skimage.morphology.binary_dilation(boundary)
            dst[boundary] = (200, 200, 200)
        except ImportError:
            pass

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
        dst = draw_module.rectangle(
            dst,
            aabb1,
            aabb2,
            outline=color_cls,
            width=line_width,
        )

        if caption is not None:
            for loc in ["lt+", "lt"]:
                y1, x1, y2, x2 = draw_module.text_in_rectangle_aabb(
                    src=dst,
                    loc=loc,
                    text=caption,
                    size=font_size,
                    aabb1=aabb1,
                    aabb2=aabb2,
                    font_path=font_path,
                )
                if (
                    y1 >= 0
                    and x1 >= 0
                    and y2 < dst.shape[0]
                    and x2 < dst.shape[1]
                ):
                    break
            dst = draw_module.text_in_rectangle(
                src=dst,
                loc=loc,
                text=caption,
                size=font_size,
                background=color_cls,
                aabb1=aabb1,
                aabb2=aabb2,
                font_path=font_path,
            )
    return dst
