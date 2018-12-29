import numpy as np

from . import draw as draw_module
from . import label as label_module


def instances2rgb(
    src,
    bboxes=None,
    labels=None,
    captions=None,
    font_size=25,
    line_width=5,
):
    assert all(l >= 0 for l in labels)
    assert len(bboxes) == len(labels) == len(captions)

    colormap = label_module.label_colormap()
    colormap = (colormap * 255).round().astype(np.uint8)

    dst = src
    for bbox, label, caption in zip(bboxes, labels, captions):
        color = tuple(colormap[label])
        y1, x1, y2, x2 = bbox

        aabb1 = np.array([y1, x1], dtype=int)
        aabb2 = np.array([y2, x2], dtype=int)
        dst = draw_module.rectangle(
            dst,
            aabb1,
            aabb2,
            color=color,
            width=line_width,
        )

        height, width = draw_module.text_size(text=caption, size=font_size)
        dst = draw_module.rectangle(
            dst,
            aabb1,
            aabb1 + [height, width],
            color=color,
            fill=color,
        )

        dst = draw_module.text(
            dst,
            (y1, x1),
            text=caption,
            color=(255, 255, 255),
            size=font_size,
        )
    return dst
