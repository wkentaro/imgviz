import warnings

import matplotlib.cm
import numpy as np
import PIL.Image

from . import draw


def rgb2gray(rgb):
    # type: (np.ndarray) -> np.ndarray
    assert rgb.ndim == 3
    gray = PIL.Image.fromarray(rgb)
    gray = gray.convert('L')
    gray = np.asarray(gray)
    return gray


def gray2rgb(gray):
    # type: (np.ndarray) -> np.ndarray
    assert gray.ndim == 2
    rgb = gray[:, :, None].repeat(3, axis=2)
    return rgb


def rgb2rgba(rgb):
    # type: (np.ndarray) -> np.ndarray
    assert rgb.ndim == 3
    a = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack((rgb, a))
    return rgba


def depth2rgb(
    depth,
    min_value=None,
    max_value=None,
    colormap='jet',
):
    if min_value is None:
        min_value = np.nanmin(depth)
    if max_value is None:
        max_value = np.nanmax(depth)

    if np.isinf(min_value) or np.isinf(max_value):
        warnings.warn('Min or max value for depth colorization is inf.')
    if max_value == min_value:
        eps = np.finfo(depth.dtype).eps
        max_value += eps
        min_value -= eps

    rgb = np.zeros(depth.shape, dtype=float)

    isnan = np.isnan(depth)
    rgb[~isnan] = 1. * (depth[~isnan] - min_value) / (max_value - min_value)

    colormap_func = getattr(matplotlib.cm, colormap)
    rgb = colormap_func(rgb)[:, :, :3]
    rgb = (rgb * 255).round().astype(np.uint8)
    rgb[isnan] = (0, 0, 0)

    return rgb


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

    if label_names is None:
        return res

    def get_fg_color(color):
        intensity = rgb2gray(color.reshape(1, 1, 3)).sum()
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
        height, width = draw.text_size(text, size=font_size)
        color = get_fg_color(res[y, x])
        res = draw.text(
            res,
            position=(y - height // 2, x - width // 2),
            text=text,
            color=color,
            size=font_size,
        )

    return res
