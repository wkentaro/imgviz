import numpy as np
import PIL.Image
import pyglet


def imread(filename):
    # type: (str) -> np.ndarray
    return np.asarray(PIL.Image.open(filename))


def imsave(filename, arr):
    # type: (str, np.ndarray) -> None
    return PIL.Image.fromarray(arr).save(filename)


# -----------------------------------------------------------------------------


def pyglet_imshow(image):
    # type: (np.ndarray) -> None
    image = PIL.Image.fromarray(image)
    try:
        image = pyglet.image.ImageData(
            width=image.width,
            height=image.height,
            format=image.mode,
            data=image.tobytes(),
            pitch=- image.width * len(image.mode),
        )
    except pyglet.canvas.xlib.NoSuchDisplayException:
        return

    window = pyglet.window.Window(
        width=image.width,
        height=image.height,
        # resizable=True,
    )
    sprite = pyglet.sprite.Sprite(image)

    @window.event
    def on_draw():
        window.clear()
        sprite.draw()

    @window.event()
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.close()

    # @window.event()
    # def on_resize(width, height):
    #     scale_w = 1. * width / image.width
    #     scale_h = 1. * height / image.height
    #
    #     image_scale = min(scale_w, scale_h)
    #     image_width = int(round(image.width * sprite.scale))
    #     image_height = int(round(image.height * sprite.scale))
    #
    #     sprite.update(
    #         x=(width - image_width) / 2.0,
    #         y=(height - image_height) / 2.0,
    #         scale=image_scale,
    #     )


def pyglet_run():
    try:
        return pyglet.app.run()
    except (pyglet.canvas.xlib.NoSuchDisplayException, TypeError):
        return
