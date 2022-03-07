import numpy as np
import PIL.Image
import PIL.ImageDraw

from .. import utils


def line(src, yx, fill, width=1):
    dst = utils.numpy_to_pillow(src)
    line_(img=dst, yx=yx, fill=fill, width=width)
    return utils.pillow_to_numpy(dst)


def line_(img, yx, fill, width=1):
    assert isinstance(img, PIL.Image.Image)
    yx = np.asarray(yx)
    assert yx.ndim == 2
    assert yx.shape[1] == 2
    fill = tuple(fill)

    draw = PIL.ImageDraw.Draw(img)

    xy = yx[:, ::-1]
    xy = xy.flatten().tolist()
    draw.line(xy, fill=fill, width=width)
