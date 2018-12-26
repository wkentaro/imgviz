import numpy as np
import PIL.Image
import PIL.ImageDraw


def rectangle(img, aabb1, aabb2, color, fill=None, width=0):
    img_pil = PIL.Image.fromarray(img)
    draw = PIL.ImageDraw.ImageDraw(img_pil)

    y1, x1 = aabb1
    y2, x2 = aabb2
    draw.rectangle(xy=(x1, y1, x2, y2), fill=fill, outline=color, width=width)

    return np.asarray(img_pil)
