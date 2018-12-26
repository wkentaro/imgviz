import numpy as np
import PIL.Image
import PIL.ImageDraw


def rectangle(src, aabb1, aabb2, color, fill=None, width=0):
    color = tuple(color)

    src_pil = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.ImageDraw(src_pil)

    y1, x1 = aabb1
    y2, x2 = aabb2
    draw.rectangle(xy=(x1, y1, x2, y2), fill=fill, outline=color, width=width)

    dst = np.asarray(src_pil)
    return dst
