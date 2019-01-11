import numpy as np

from . import color as color_module
from . import draw as draw_module
from . import label as label_module


def mask_to_bbox(masks):
    bboxes = np.zeros((len(masks), 4), dtype=float)
    for i, mask in enumerate(masks):
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
    alpha=0.7,
):
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8
    assert image.ndim == 3

    assert all(l >= 0 for l in labels)

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

    colormap = label_module.label_colormap()
    colormap = (colormap * 255).round().astype(np.uint8)

    dst = image
    image_gray = color_module.gray2rgb(color_module.rgb2gray(image))

    for instance_id in range(n_instance):
        mask = masks[instance_id]

        if mask is None:
            continue

        color_ins = colormap[1:][instance_id % len(colormap[1:])]

        maskviz = mask[:, :, None] * color_ins.astype(float)
        dst = dst.copy()
        dst[mask] = (
            (1 - alpha) * image_gray[mask].astype(float) +
            alpha * maskviz[mask]
        )

        try:
            import skimage.segmentation
            boundary = skimage.segmentation.find_boundaries(
                mask, connectivity=2
            )
            dst[boundary] = (200, 200, 200)
        except ImportError:
            pass

    for instance_id in range(n_instance):
        bbox = bboxes[instance_id]
        label = labels[instance_id]
        caption = captions[instance_id]

        color_cls = colormap[label % len(colormap)]

        y1, x1, y2, x2 = bbox
        aabb1 = np.array([y1, x1], dtype=int)
        aabb2 = np.array([y2, x2], dtype=int)
        dst = draw_module.rectangle(
            dst,
            aabb1,
            aabb2,
            color=tuple(color_cls),
            width=line_width,
        )

        if caption is not None:
            height, width = draw_module.text_size(text=caption, size=font_size)
            dst = draw_module.rectangle(
                dst,
                aabb1,
                aabb1 + [height, width],
                color=tuple(color_cls),
                fill=tuple(color_cls),
            )
            dst = draw_module.text(
                dst,
                (y1, x1),
                text=caption,
                color=(255, 255, 255),
                size=font_size,
            )
    return dst
