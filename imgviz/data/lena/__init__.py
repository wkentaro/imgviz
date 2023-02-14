import os.path as osp

from ..._io import imread

here = osp.dirname(osp.abspath(__file__))


def lena():
    image_file = osp.join(here, "lena.png")
    image = imread(image_file)
    return image
