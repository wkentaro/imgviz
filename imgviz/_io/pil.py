import PIL.Image


def pil_imshow(image):
    PIL.Image.fromarray(image).show()
