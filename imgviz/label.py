import numpy as np

from . import color as color_module
from . import draw as draw_module


def label_colormap(n_label=256):
    # type: (int) -> np.ndarray
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((n_label, 3))
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
    cmap = cmap.astype(np.float32) / 255
    return cmap


def label2rgb(
    label,
    img=None,
    alpha=0.5,
    label_names=None,
    n_labels=None,
    font_size=30,
    thresh_suppress=0,
):
    if n_labels is None:
        if label_names:
            n_labels = len(label_names)
        else:
            n_labels = np.max(label) + 1  # +1 for background label 0
    elif label_names:
        assert n_labels == len(label_names)
        assert np.max(label) < n_labels

    colormap = label_colormap(n_labels)
    colormap = (colormap * 255).astype(np.uint8)

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

    def get_fg_color(color):
        intensity = color_module.rgb2gray(color.reshape(1, 1, 3)).sum()
        if intensity > 170:
            return (0, 0, 0)
        return (255, 255, 255)

    def center_of_mass(mask):
        assert mask.ndim == 2 and mask.dtype == bool
        mask = 1. * mask / mask.sum()
        dx = np.sum(mask, 0)
        dy = np.sum(mask, 1)
        cx = np.sum(dx * np.arange(mask.shape[1]))
        cy = np.sum(dy * np.arange(mask.shape[0]))
        return cy, cx

    for l in np.unique(label):
        if l == -1:
            continue  # unlabeled

        mask = label == l
        if 1. * mask.sum() / mask.size < thresh_suppress:
            continue
        y, x = np.array(center_of_mass(mask), dtype=int)

        if label[y, x] != l:
            Y, X = np.where(mask)
            point_index = np.random.randint(0, len(Y))
            y, x = Y[point_index], X[point_index]

        text = label_names[l]
        height, width = draw_module.text_size(text, size=font_size)
        color = get_fg_color(res[y, x])
        res = draw_module.text(
            res,
            yx=(y - height // 2, x - width // 2),
            text=text,
            color=color,
            size=font_size,
        )

    return res
