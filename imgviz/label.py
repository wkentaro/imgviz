import numpy as np

from . import color as color_module
from . import draw as draw_module


def label_colormap(n_label=256, value=None):

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    if value is not None:
        hsv = color_module.rgb2hsv(cmap.reshape(1, -1, 3))
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        cmap = color_module.hsv2rgb(hsv).reshape(-1, 3)
    return cmap


def label2rgb(
    label,
    img=None,
    alpha=0.5,
    label_names=None,
    font_size=30,
    thresh_suppress=0,
    colormap=None,
    loc='centroid',
):
    if colormap is None:
        colormap = label_colormap()

    res = colormap[label]

    np.random.seed(1234)

    mask_unlabeled = label < 0
    res[mask_unlabeled] = \
        np.random.random(size=(mask_unlabeled.sum(), 3)) * 255

    if img is not None:
        if img.ndim == 2:
            img = color_module.gray2rgb(img)
        res = (1 - alpha) * img.astype(float) + alpha * res.astype(float)
        res = np.clip(res.round(), 0, 255).astype(np.uint8)

    if label_names is None:
        return res

    if loc == 'centroid':
        for l in np.unique(label):
            if l == -1:
                continue  # unlabeled

            mask = label == l
            if 1. * mask.sum() / mask.size < thresh_suppress:
                continue
            y, x = np.array(_center_of_mass(mask), dtype=int)

            if label[y, x] != l:
                Y, X = np.where(mask)
                point_index = np.random.randint(0, len(Y))
                y, x = Y[point_index], X[point_index]

            text = label_names[l]
            height, width = draw_module.text_size(text, size=font_size)
            color = color_module.get_fg_color(res[y, x])
            res = draw_module.text(
                res,
                yx=(y - height // 2, x - width // 2),
                text=text,
                color=color,
                size=font_size,
            )
    elif loc in ['rb', 'lt']:
        unique_labels = np.unique(label)
        unique_labels = unique_labels[unique_labels != -1]
        text_sizes = np.array([
            draw_module.text_size(label_names[l], font_size)
            for l in unique_labels
        ])
        text_height, text_width = text_sizes.max(axis=0)
        legend_height = text_height * len(unique_labels) + 5
        legend_width = text_width + 40

        height, width = label.shape[:2]
        legend = np.zeros((height, width, 3), dtype=np.uint8)
        if loc == 'rb':
            aabb2 = np.array([height - 5, width - 5], dtype=float)
            aabb1 = aabb2 - (legend_height, legend_width)
        elif loc == 'lt':
            aabb1 = np.array([5, 5], dtype=float)
            aabb2 = aabb1 + (legend_height, legend_width)
        else:
            raise ValueError('unexpected loc: {}'.format(loc))
        legend = draw_module.rectangle(
            legend, aabb1, aabb2, fill=(255, 255, 255))

        alpha = 0.5
        y1, x1 = aabb1.round().astype(int)
        y2, x2 = aabb2.round().astype(int)
        res[y1:y2, x1:x2] = \
            alpha * res[y1:y2, x1:x2] + alpha * legend[y1:y2, x1:x2]

        for i, l in enumerate(unique_labels):
            box_aabb1 = aabb1 + (i * text_height + 5, 5)
            box_aabb2 = box_aabb1 + (text_height - 10, 20)
            res = draw_module.rectangle(
                res,
                aabb1=box_aabb1,
                aabb2=box_aabb2,
                fill=colormap[l]
            )
            res = draw_module.text(
                res,
                yx=aabb1 + (i * text_height, 30),
                text=label_names[l],
                size=font_size,
            )
    else:
        raise ValueError('unsupported loc: {}'.format(loc))

    return res


def _center_of_mass(mask):
    assert mask.ndim == 2 and mask.dtype == bool
    mask = 1. * mask / mask.sum()
    dx = np.sum(mask, 0)
    dy = np.sum(mask, 1)
    cx = np.sum(dx * np.arange(mask.shape[1]))
    cy = np.sum(dy * np.arange(mask.shape[0]))
    return cy, cx
