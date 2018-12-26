import numpy as np

from .resize import resize


def centerize(src, shape, cval=None, return_mask=False):
    """Centerize image for specified image size

    Parameters
    ----------
    src: numpy.ndarray
        Image to centerize
    shape: tuple of int
        Image shape (height, width) or (height, width, channel)
    cval: numpy.ndarray
        Color to be filled in the blank.
    return_mask: numpy.ndarray
        Mask for centerized image.

    Returns
    -------
    dst: numpy.ndarray
        Centerized image.
    """
    if src.shape[:2] == shape[:2]:
        if return_mask:
            return src, np.ones(shape[:2], dtype=bool)
        else:
            return src

    if len(shape) != src.ndim:
        shape = list(shape) + [src.shape[2]]

    dst = np.zeros(shape, dtype=src.dtype)
    if cval:
        dst[:, :] = cval

    src_h, src_w = src.shape[:2]
    scale_h, scale_w = 1. * shape[0] / src_h, 1. * shape[1] / src_w
    scale = min(scale_h, scale_w)
    dst_h, dst_w = int(round(src_h * scale)), int(round(src_w * scale))
    src = resize(src, height=dst_h, width=dst_w)

    ph, pw = 0, 0
    h, w = src.shape[:2]
    dst_h, dst_w = shape[:2]
    if h < dst_h:
        ph = (dst_h - h) // 2
    if w < dst_w:
        pw = (dst_w - w) // 2
    dst[ph:ph + h, pw:pw + w] = src

    if return_mask:
        mask = np.zeros(shape[:2], dtype=bool)
        mask[ph:ph + h, pw:pw + w] = True
        return dst, mask
    else:
        return dst
