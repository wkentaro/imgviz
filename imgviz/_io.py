import numpy as np
import PIL.Image

try:
    import pyglet
except ImportError:
    pyglet = None


def imread(filename):
    # type: (str) -> np.ndarray
    return np.asarray(PIL.Image.open(filename))


def imsave(filename, arr):
    # type: (str, np.ndarray) -> None
    return PIL.Image.fromarray(arr).save(filename)


# -----------------------------------------------------------------------------
# pyglet


def pyglet_imshow(image):
    # type: (np.ndarray) -> None
    image = PIL.Image.fromarray(image)
    image = pyglet.image.ImageData(
        width=image.width,
        height=image.height,
        format=image.mode,
        data=image.tobytes(),
        pitch=- image.width * len(image.mode),
    )

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
    return pyglet.app.run()


# -----------------------------------------------------------------------------
# matplotlib


def pyplot_fig2arr(fig):
    fig.canvas.draw()
    arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    arr = arr.reshape((height, width, 3))
    return arr
