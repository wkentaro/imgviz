import types

import numpy as np
import PIL.Image

try:
    import pyglet
except ImportError:
    pyglet = None


def check_pyglet_available():
    if pyglet is None:
        raise ImportError(
            'pyglet is not installed, run following: pip install pyglet'
        )


def pyglet_run():
    check_pyglet_available()

    return pyglet.app.run()


def pyglet_imshow(image, caption=None, **kwargs):
    check_pyglet_available()

    if isinstance(image, types.GeneratorType):
        _pyglet_imshow_generator(image, caption=caption, **kwargs)
    elif isinstance(image, np.ndarray):
        _pyglet_imshow_ndarray(image, caption=caption)
    else:
        _pyglet_imshow_list(image, caption=caption, **kwargs)


def _pyglet_imshow_list(images, caption, interval=0.5):
    index = 0

    image = _ndarray_to_imagedata(images[index])
    window, sprite = _window_and_sprite(image, caption=caption)

    window.index = index
    window.play = False

    @window.event
    def on_draw():
        window.clear()
        sprite.draw()

    def usage():
        print('Usage: ')
        print('  h: show help')
        print('  q: close window')
        print('  n: next image')
        print('  p: previous image')
        print('  s: toggle play')

    def play_callback(dt):
        if window.play:
            if window.index == len(images) - 1:
                print("Press 'q' to quit")
                window.play = False
            window.index = min(window.index + 1, len(images) - 1)
            sprite.image = _ndarray_to_imagedata(images[window.index])

    pyglet.clock.schedule_interval(play_callback, interval)

    @window.event()
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.close()
            elif symbol == pyglet.window.key.N:
                window.index = min(window.index + 1, len(images) - 1)
                sprite.image = _ndarray_to_imagedata(images[window.index])
            elif symbol == pyglet.window.key.P:
                window.index = max(window.index - 1, 0)
                sprite.image = _ndarray_to_imagedata(images[window.index])
            elif symbol == pyglet.window.key.S:
                window.play = not window.play
            elif symbol == pyglet.window.key.H:
                usage()
            else:
                print("Press 'h' to show help")


def _pyglet_imshow_generator(images, caption, interval=0.5):
    image = _ndarray_to_imagedata(next(images))
    window, sprite = _window_and_sprite(image, caption=caption)

    window.play = False

    @window.event
    def on_draw():
        window.clear()
        sprite.draw()

    def usage():
        print('Usage: ')
        print('  h: show help')
        print('  q: close window')
        print('  n: next image')
        print('  s: toggle play')

    def play_callback(dt):
        if window.play:
            try:
                sprite.image = _ndarray_to_imagedata(next(images))
            except StopIteration:
                print("Press 'q' to quit")
                window.play = False

    pyglet.clock.schedule_interval(play_callback, interval)

    @window.event()
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.close()
            elif symbol == pyglet.window.key.N:
                try:
                    sprite.image = _ndarray_to_imagedata(next(images))
                except StopIteration:
                    print("Press 'q' to quit")
            elif symbol == pyglet.window.key.S:
                window.play = not window.play
            elif symbol == pyglet.window.key.H:
                usage()
            else:
                print("Press 'h' to show help")


def _pyglet_imshow_ndarray(image, caption):
    image = _ndarray_to_imagedata(image)
    window, sprite = _window_and_sprite(image, caption=caption)

    @window.event
    def on_draw():
        window.clear()
        sprite.draw()

    def usage():
        print('Usage: ')
        print('  h: show help')
        print('  q: close window')

    @window.event()
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.close()
            elif symbol == pyglet.window.key.H:
                usage()
            else:
                print("Press 'h' to show help")


def _window_and_sprite(image, caption):
    window = pyglet.window.Window(
        width=image.width,
        height=image.height,
        caption=caption,
    )
    sprite = pyglet.sprite.Sprite(image)
    return window, sprite


def _ndarray_to_imagedata(image):
    image = PIL.Image.fromarray(image)
    image = pyglet.image.ImageData(
        width=image.width,
        height=image.height,
        format=image.mode,
        data=image.tobytes(),
        pitch=- image.width * len(image.mode),
    )
    return image
