import os.path as osp

import matplotlib
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


def rectangle(src, aabb1, aabb2, color, fill=None, width=0):
    color = tuple(color)

    src_pil = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.ImageDraw(src_pil)

    y1, x1 = aabb1
    y2, x2 = aabb2
    draw.rectangle(xy=(x1, y1, x2, y2), fill=fill, outline=color, width=width)

    dst = np.asarray(src_pil)
    return dst


def _get_font(size):
    fonts_path = osp.join(
        osp.dirname(matplotlib.__file__), 'mpl-data/fonts/ttf'
    )
    font_path = osp.join(fonts_path, 'DejaVuSans.ttf')
    font = PIL.ImageFont.truetype(font=font_path, size=size)
    return font


def text_size(text, size):
    font = _get_font(size)
    width, height = font.getsize(text)
    return height, width


def text(src, position, text, color, size):
    src_pil = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.ImageDraw(src_pil)

    y1, x1 = position
    color = tuple(color)
    font = _get_font(size=size)
    draw.text(xy=(x1, y1), text=text, fill=color, font=font)

    dst = np.asarray(src_pil)
    return dst
