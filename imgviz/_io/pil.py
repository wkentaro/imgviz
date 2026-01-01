from .. import _utils


def pil_imshow(image):
    _utils.numpy_to_pillow(image).show()
