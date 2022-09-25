try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from .. import utils


def ellipse(src, yx1, yx2, fill=None, outline=None, width=0):
    dst = utils.numpy_to_pillow(src)
    ellipse_(
        img=dst, yx1=yx1, yx2=yx2, fill=fill, outline=outline, width=width
    )
    return utils.pillow_to_numpy(dst)


def ellipse_(img, yx1, yx2, fill=None, outline=None, width=0):
    if isinstance(fill, Iterable):
        fill = tuple(fill)
    if isinstance(outline, Iterable):
        outline = tuple(outline)

    draw = PIL.ImageDraw.Draw(img)

    y1, x1 = yx1
    y2, x2 = yx2

    draw.ellipse([x1, y1, x2, y2], fill=fill, outline=outline, width=width)
