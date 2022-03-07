import numpy as np
import PIL.Image


def pillow_to_numpy(img):
    img_numpy = np.asarray(img)
    if not img_numpy.flags.writeable:
        img_numpy = np.array(img)
    return img_numpy


def numpy_to_pillow(img, mode=None):
    img_pillow = PIL.Image.fromarray(img, mode=mode)
    return img_pillow
