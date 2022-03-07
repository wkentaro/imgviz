import collections

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


def ellipse(src, yx1, yx2, fill=None, outline=None, width=0):
    if isinstance(fill, collections.abc.Iterable):
        fill = tuple(fill)
    if isinstance(outline, collections.abc.Iterable):
        outline = tuple(outline)

    dst = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.Draw(dst)

    y1, x1 = yx1
    y2, x2 = yx2

    draw.ellipse([x1, y1, x2, y2], fill=fill, outline=outline, width=width)

    return np.array(dst)
