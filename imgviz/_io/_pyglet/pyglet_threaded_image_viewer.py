from __future__ import print_function

import sys
import threading

import pyglet

from .pyglet_imshow import _ndarray_to_imagedata


class PygletThreadedImageViewer(pyglet.window.Window):

    """Image viewer with threading."""

    def __init__(
        self, play=True, interval=0.5, height=None, width=None, resizable=True
    ):
        """Initialize the image viewer.

        Parameters
        ----------
        play: bool
            If True, it shows automatically for each `imshow()`.
        interval: float
            Interval in seconds for each `imshow()` call.

        .. seealso:: :class:`pyglet.window.Window`

        """
        self._play = play
        self._next = False

        self._updated_at = 0
        self._interval = interval
        self._added_at = -interval

        self.sprite = None

        if isinstance(height, float):
            height = int(round(height))
        if isinstance(width, float):
            width = int(round(width))

        self.lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._init_and_start_app,
            kwargs=dict(height=height, width=width, resizable=resizable),
        )
        self.thread.daemon = True  # terminate when main thread exit
        self.thread.start()
        pyglet.clock.schedule_interval(self.on_update, 1.0 / 100)

    def imshow(self, image):
        """Update image on the viewer.

        Parameters
        ----------
        image: numpy.ndarray
            The image to show on the viewer.

        """
        imagedata = _ndarray_to_imagedata(image)

        width, height = self.get_size()
        scale = min(
            float(width) / imagedata.width,
            float(height) / imagedata.height,
        )
        position_x = (width - imagedata.width * scale) / 2
        position_y = (height - imagedata.height * scale) / 2

        with self.lock:
            if self.sprite is None:
                self.sprite = pyglet.sprite.Sprite(imagedata)
            else:
                self.sprite.image = imagedata
            self.sprite.update(x=position_x, y=position_y, scale=scale)
            self._added_at = self._updated_at

    def wait(self):
        while True:
            with self.lock:
                if self._play and self._updated_at >= (
                    self._added_at + self._interval
                ):
                    break
                elif self._next:
                    self._next = False
                    break

    def _init_and_start_app(self, **kwargs):
        super(PygletThreadedImageViewer, self).__init__(**kwargs)
        pyglet.app.run()

    def on_draw(self):
        self.clear()
        self.sprite.draw()

    def on_update(self, dt):
        if self.sprite is None:
            return
        with self.lock:
            self.on_draw()
            self._updated_at += dt

    def on_close(self):
        pyglet.clock.unschedule(self.on_update)
        with self.lock:
            super(PygletThreadedImageViewer, self).on_close()

    def wait_key(self):
        with self.lock:
            self._waiting_key = True
            self._key = None

        key = None
        while True:
            with self.lock:
                if self._key:
                    key = self._key
                    break

        with self.lock:
            self._waiting_key = False

        return key

    def on_key_press(self, symbol, modifiers):
        with self.lock:
            if self._waiting_key:
                self._key = symbol, modifiers
                return

        def usage():
            message = """Usage:
\th: show help
\tq: close window
\tn: next image
\ts: toggle play"""
            print(message, file=sys.stderr)

        if symbol == pyglet.window.key.H:
            usage()
        elif symbol == pyglet.window.key.Q:
            self.on_close()
        elif symbol == pyglet.window.key.S:
            with self.lock:
                self._play = not self._play
        elif symbol == pyglet.window.key.N:
            with self.lock:
                self._next = True
