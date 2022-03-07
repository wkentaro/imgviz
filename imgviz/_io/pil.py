from .. import utils


def pil_imshow(image):
    utils.numpy_to_pillow(image).show()
