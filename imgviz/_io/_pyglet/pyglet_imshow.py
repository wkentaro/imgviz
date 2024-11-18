import sys
import types

import numpy as np
import PIL.Image

from ... import utils
from .base import check_pyglet_available
from .pyglet_run import pyglet_run


def pyglet_imshow(image, caption=None, interval=0.5, keymap=None, hook=None, run=False):
    """Show image with pyglet.

    Parameters
    ----------
    image: numpy.ndarray or list of numpy.ndarray or iterator of numpy.array
        Image or images to show.
    caption: str, optional
        Caption for pyglet window.
    interval: float, optional
        Interval for list or iterator of images. Default is 0.5.
    keymap: dict, optional
        Key mappings for key and function.
    hook: callable, optional
        Hook function called after each image update.
    run: bool, optional
        If True, pyglet_run is called after imshow. Default is False.

    Returns
    -------
    None

    """
    if isinstance(image, types.GeneratorType):
        _pyglet_imshow_generator(
            image,
            caption=caption,
            interval=interval,
            keymap=keymap,
            hook=hook,
        )
    elif isinstance(image, np.ndarray):
        _pyglet_imshow_ndarray(image, caption=caption, keymap=keymap, hook=hook)
    else:
        _pyglet_imshow_list(
            image,
            caption=caption,
            interval=interval,
            keymap=keymap,
            hook=hook,
        )

    if run:
        pyglet_run()


def _pyglet_imshow_list(images, caption=None, interval=0.5, keymap=None, hook=None):
    pyglet = check_pyglet_available()

    def usage():
        print("Usage: ", file=sys.stderr)
        if keymap is not None:
            for key, function in keymap.items():
                print("  {}: {}".format(key, function))
        print("  h: show help", file=sys.stderr)
        print("  q: close window", file=sys.stderr)
        print("  n: next image", file=sys.stderr)
        print("  p: previous image", file=sys.stderr)
        print("  s: toggle play", file=sys.stderr)

    usage()

    if isinstance(images[0], np.ndarray):
        max_image_height, max_image_width = np.max(
            [image.shape[:2] for image in images], axis=0
        )
    elif isinstance(images[0], PIL.Image.Image):
        max_image_width, max_image_height = np.max(
            [image.size for image in images], axis=0
        )
    elif hook is None:
        raise ValueError(
            "hook must be specified when images is not list of "
            "numpy.ndarray or PIL.Image.Image"
        )
    else:
        image = hook(images[0])
        if not isinstance(image, (np.ndarray, PIL.Image.Image)):
            raise TypeError("hook must return numpy.ndarray or PIL.Image.Image")
        max_image_height, max_image_width = image.shape[:2]
    aspect_ratio = max_image_width / max_image_height

    index = 0

    window = _initialize_window(caption=caption, aspect_ratio=aspect_ratio)
    window.index = index
    window.play = False

    image = _convert_to_imagedata(
        hook(images[window.index]) if hook else images[window.index]
    )
    sprite = pyglet.sprite.Sprite(image)

    def _post_image_update():
        _centerize_sprite_in_window(sprite, window)
        window.set_caption(
            "{} {}/{}".format(images[window.index], window.index + 1, len(images))
        )
        print(
            "{} {}/{}".format(images[window.index], window.index + 1, len(images)),
            file=sys.stderr,
        )

    _post_image_update()

    @window.event
    def on_draw():
        window.clear()
        sprite.draw()

    def play_callback(dt):
        if window.play:
            if window.index == len(images) - 1:
                print("Press 'q' to quit", file=sys.stderr)
                window.play = False
            window.index = min(window.index + 1, len(images) - 1)
            sprite.image = _convert_to_imagedata(
                hook(images[window.index]) if hook else images[window.index]
            )
            _centerize_sprite_in_window(sprite, window)

    pyglet.clock.schedule_interval(play_callback, interval)

    @window.event
    def on_key_press(symbol, modifiers):
        if keymap is not None:
            for key, function in keymap.items():
                if symbol == getattr(pyglet.window.key, key.upper()):
                    function(images, window.index)
                    return
        if symbol == pyglet.window.key.Q:
            window.close()
        elif symbol == pyglet.window.key.N:
            if window.index + 1 <= len(images) - 1:
                window.index += 1
                sprite.image = _convert_to_imagedata(
                    hook(images[window.index]) if hook else images[window.index]
                )
                _post_image_update()
        elif symbol == pyglet.window.key.P:
            if window.index - 1 >= 0:
                window.index -= 1
                sprite.image = _convert_to_imagedata(
                    hook(images[window.index]) if hook else images[window.index]
                )
                _post_image_update()
        elif symbol == pyglet.window.key.S:
            window.play = not window.play
        elif symbol == pyglet.window.key.H:
            usage()
        else:
            print("Press 'h' to show help", file=sys.stderr)


def _pyglet_imshow_generator(
    images, caption=None, interval=0.5, keymap=None, hook=None
):
    if keymap is not None:
        raise NotImplementedError

    pyglet = check_pyglet_available()

    image = hook(next(images)) if hook else images
    if not isinstance(image, (np.ndarray, PIL.Image.Image)):
        raise TypeError("hook must return numpy.ndarray or PIL.Image.Image")

    aspect_ratio = image.shape[1] / image.shape[0]  # width / height
    window = _initialize_window(caption=caption, aspect_ratio=aspect_ratio)

    image = _convert_to_imagedata(image)
    sprite = pyglet.sprite.Sprite(image)
    _centerize_sprite_in_window(sprite, window)

    window.play = False

    @window.event
    def on_draw():
        window.clear()
        sprite.draw()

    def usage():
        print("Usage: ", file=sys.stderr)
        print("  h: show help", file=sys.stderr)
        print("  q: close window", file=sys.stderr)
        print("  n: next image", file=sys.stderr)
        print("  p: previous image", file=sys.stderr)
        print("  s: toggle play", file=sys.stderr)

    def play_callback(dt):
        if window.play:
            try:
                sprite.image = _convert_to_imagedata(next(images))
                _centerize_sprite_in_window(sprite, window)
            except StopIteration:
                print("Press 'q' to quit", file=sys.stderr)
                window.play = False

    pyglet.clock.schedule_interval(play_callback, interval)

    @window.event()
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.Q:
            window.close()
        elif symbol == pyglet.window.key.N:
            try:
                sprite.image = _convert_to_imagedata(
                    hook(next(images)) if hook else next(images)
                )
                _centerize_sprite_in_window(sprite, window)
            except StopIteration:
                print("Press 'q' to quit", file=sys.stderr)
        elif symbol == pyglet.window.key.S:
            window.play = not window.play
        elif symbol == pyglet.window.key.H:
            usage()
        else:
            print("Press 'h' to show help", file=sys.stderr)


def _pyglet_imshow_ndarray(image, caption=None, keymap=None, hook=None):
    if keymap is not None:
        raise NotImplementedError

    pyglet = check_pyglet_available()

    image = hook(image) if hook else image
    if not isinstance(image, (np.ndarray, PIL.Image.Image)):
        raise TypeError("hook must return numpy.ndarray or PIL.Image.Image")
    aspect_ratio = image.shape[1] / image.shape[0]  # width / height
    window = _initialize_window(caption=caption, aspect_ratio=aspect_ratio)

    image = _convert_to_imagedata(image)
    sprite = pyglet.sprite.Sprite(image)
    _centerize_sprite_in_window(sprite, window)

    @window.event
    def on_draw():
        window.clear()
        sprite.draw()

    def usage():
        print("Usage: ", file=sys.stderr)
        print("  h: show help", file=sys.stderr)
        print("  q: close window", file=sys.stderr)

    @window.event()
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.Q:
            window.close()
        elif symbol == pyglet.window.key.H:
            usage()
        else:
            print("Press 'h' to show help", file=sys.stderr)


def _initialize_window(caption, aspect_ratio):
    pyglet = check_pyglet_available()

    display = pyglet.canvas.Display()
    screen = display.get_default_screen()

    max_window_width = int(round(screen.width * 0.75))
    max_window_height = int(round(screen.height * 0.75))

    # try to fit the image into the screen
    if aspect_ratio > 1:  # width > height
        window_width = max_window_width
        window_height = int(round(window_width / aspect_ratio))
    else:
        window_height = max_window_height
        window_width = int(round(window_height * aspect_ratio))

    # if still too large, shrink it
    if window_width > max_window_width:
        window_width = max_window_width
        window_height = int(round(window_width / aspect_ratio))
    if window_height > max_window_height:
        window_height = max_window_height
        window_width = int(round(window_height * aspect_ratio))

    window = pyglet.window.Window(
        width=window_width,
        height=window_height,
        caption=caption,
    )
    return window


def _centerize_sprite_in_window(sprite, window):
    scale_x = 0.95 * window.width / sprite.image.width
    scale_y = 0.95 * window.height / sprite.image.height
    scale = min(scale_x, scale_y)

    width = sprite.image.width * scale
    height = sprite.image.height * scale
    x = (window.width - width) / 2.0
    y = (window.height - height) / 2.0

    sprite.update(x=x, y=y, scale=scale)


def _convert_to_imagedata(image):
    pyglet = check_pyglet_available()

    if isinstance(image, np.ndarray):
        image = utils.numpy_to_pillow(image)
    elif isinstance(image, PIL.Image.Image):
        pass
    else:
        raise ValueError

    image = image.convert("RGB")

    kwargs = dict(
        width=image.width,
        height=image.height,
        data=image.tobytes(),
        pitch=-image.width * len(image.mode),
    )
    if hasattr(pyglet, "__version__") and pyglet.__version__[0] == "2":
        kwargs["fmt"] = image.mode
    else:
        kwargs["format"] = image.mode
    image = pyglet.image.ImageData(**kwargs)
    return image
