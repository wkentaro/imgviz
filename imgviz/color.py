import numpy as np
import PIL.Image


def rgb2gray(rgb):
    gray = PIL.Image.fromarray(rgb)
    gray = gray.convert('L')
    gray = np.asarray(gray)
    return gray


def gray2rgb(gray):
    rgb = gray[:, :, None].repeat(3, axis=2)
    return rgb
