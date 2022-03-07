import numpy as np
import PIL.Image
import PIL.ImageDraw


def line(src, yx, fill, width=1):
    fill = tuple(fill)
    yx = np.asarray(yx)
    assert yx.ndim == 2
    assert yx.shape[1] == 2

    dst = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.Draw(dst)

    xy = yx[:, ::-1]
    xy = xy.flatten().tolist()
    draw.line(xy, fill=fill, width=width)

    return np.asarray(dst)
