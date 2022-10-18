import numbers

import numpy as np

from . import color as color_module
from . import draw as draw_module
from . import utils


def label_colormap(n_label=256, value=None):
    """Label colormap.

    Parameters
    ----------
    n_labels: int
        Number of labels (default: 256).
    value: float or int
        Value scale or value of label color in HSV space.

    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.

    """

    def bitget(byteval, idx):
        shape = byteval.shape + (8,)
        return np.unpackbits(byteval).reshape(shape)[..., -1 - idx]

    i = np.arange(n_label, dtype=np.uint8)
    r = np.full_like(i, 0)
    g = np.full_like(i, 0)
    b = np.full_like(i, 0)

    i = np.repeat(i[:, None], 8, axis=1)
    i = np.right_shift(i, np.arange(0, 24, 3)).astype(np.uint8)
    j = np.arange(8)[::-1]
    r = np.bitwise_or.reduce(np.left_shift(bitget(i, 0), j), axis=1)
    g = np.bitwise_or.reduce(np.left_shift(bitget(i, 1), j), axis=1)
    b = np.bitwise_or.reduce(np.left_shift(bitget(i, 2), j), axis=1)

    cmap = np.stack((r, g, b), axis=1).astype(np.uint8)

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
    image=None,
    alpha=0.5,
    label_names=None,
    font_size=30,
    thresh_suppress=0,
    colormap=None,
    loc="rb",
    font_path=None,
):
    """Convert label to rgb.

    Parameters
    ----------
    label: numpy.ndarray, (H, W), int
        Label image.
    image: numpy.ndarray, (H, W, 3), numpy.uint8
        RGB image.
    alpha: float, or list or dict of float
        Alpha of RGB (default: 0.5).
        If given as a list or dict, it is treated as alpha for each class
        according to the index or key.
    label_names: list or dict of string
        Label id to label name.
    font_size: int
        Font size (default: 30).
    thresh_suppress: float
        Threshold of label ratio in the label image.
    colormap: numpy.ndarray, (M, 3), numpy.uint8
        Label id to color.
        By default, :func:`~imgviz.label_colormap` is used.
    loc: string
        Location of legend (default: 'rb').
        'centroid', 'lt' and 'rb' are supported.
    font_path: str
        Font path.

    Returns
    -------
    res: numpy.ndarray, (H, W, 3), numpy.uint8
        Visualized image.

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

    if isinstance(alpha, numbers.Number):
        alpha = np.array([alpha for _ in range(max_label_id + 1)])
    elif isinstance(alpha, dict):
        alpha = np.array([alpha.get(l, 0.5) for l in range(max_label_id + 1)])
    else:
        alpha = np.asarray(alpha)
        assert alpha.ndim == 1
    assert ((0 <= alpha) & (alpha <= 1)).all()
    alpha = alpha[label][:, :, None]

    if image is not None:
        if image.ndim == 2:
            image = color_module.gray2rgb(image)
        res = (1 - alpha) * image.astype(float) + alpha * res.astype(float)
        res = np.clip(res.round(), 0, 255).astype(np.uint8)

    if label_names is None:
        return res

    unique_labels = unique_labels[unique_labels != -1]
    if isinstance(label_names, dict):
        unique_labels = [l for l in unique_labels if label_names.get(l)]
    else:
        unique_labels = [l for l in unique_labels if label_names[l]]
    if len(unique_labels) == 0:
        return res

    if loc == "centroid":
        res = utils.numpy_to_pillow(res)
        for label_i in unique_labels:
            mask = label == label_i
            if 1.0 * mask.sum() / mask.size < thresh_suppress:
                continue
            y, x = np.array(_center_of_mass(mask), dtype=int)

            if label[y, x] != label_i:
                Y, X = np.where(mask)
                point_index = np.random.randint(0, len(Y))
                y, x = Y[point_index], X[point_index]

            text = label_names[label_i]
            height, width = draw_module.text_size(
                text, size=font_size, font_path=font_path
            )
            color = color_module.get_fg_color(res.getpixel((x, y)))
            draw_module.text_(
                res,
                yx=(y - height // 2, x - width // 2),
                text=text,
                color=color,
                size=font_size,
                font_path=font_path,
            )
    elif loc in ["rb", "lt"]:
        text_sizes = np.array(
            [
                draw_module.text_size(
                    label_names[l], font_size, font_path=font_path
                )
                for l in unique_labels
            ]
        )
        text_height, text_width = text_sizes.max(axis=0)
        legend_height = text_height * len(unique_labels) + 5
        legend_width = text_width + 20 + (text_height - 10)

        height, width = label.shape[:2]
        if loc == "rb":
            aabb2 = np.array([height - 5, width - 5], dtype=float)
            aabb1 = aabb2 - (legend_height, legend_width)
        elif loc == "lt":
            aabb1 = np.array([5, 5], dtype=float)
            aabb2 = aabb1 + (legend_height, legend_width)
        else:
            raise ValueError("unexpected loc: {}".format(loc))

        alpha = 0.5
        y1, x1 = aabb1.round().astype(int)
        y2, x2 = aabb2.round().astype(int)
        res[y1:y2, x1:x2] = alpha * res[y1:y2, x1:x2] + alpha * 255

        res = utils.numpy_to_pillow(res)
        for i, l in enumerate(unique_labels):
            box_aabb1 = aabb1 + (i * text_height + 5, 5)
            box_aabb2 = box_aabb1 + (text_height - 10, text_height - 10)
            draw_module.rectangle_(
                res, aabb1=box_aabb1, aabb2=box_aabb2, fill=colormap[l]
            )
            draw_module.text_(
                res,
                yx=aabb1 + (i * text_height, 10 + (text_height - 10)),
                text=label_names[l],
                size=font_size,
                font_path=font_path,
            )
    else:
        raise ValueError("unsupported loc: {}".format(loc))

    return utils.pillow_to_numpy(res)


def _center_of_mass(mask):
    assert mask.ndim == 2 and mask.dtype == bool
    mask = 1.0 * mask / mask.sum()
    dx = np.sum(mask, 0)
    dy = np.sum(mask, 1)
    cx = np.sum(dx * np.arange(mask.shape[1]))
    cy = np.sum(dy * np.arange(mask.shape[0]))
    return cy, cx
