import numpy as np
import PIL.Image
import PIL.ImageDraw

from .. import _utils


def line(src, yx, fill, width=1):
    dst = _utils.numpy_to_pillow(src)
    line_(img=dst, yx=yx, fill=fill, width=width)
    return _utils.pillow_to_numpy(dst)


def line_(img, yx, fill, width=1):
    if not isinstance(img, PIL.Image.Image):
        raise TypeError(f"img must be PIL.Image.Image, but got {type(img).__name__}")
    yx = np.asarray(yx)
    if yx.ndim != 2:
        raise ValueError(f"yx must be 2D array, but got {yx.ndim}D")
    if yx.shape[1] != 2:
        raise ValueError(f"yx.shape[1] must be 2, but got {yx.shape[1]}")
    fill = tuple(fill)

    draw = PIL.ImageDraw.Draw(img)

    xy = yx[:, ::-1]
    xy = xy.flatten().tolist()
    draw.line(xy, fill=fill, width=width)
