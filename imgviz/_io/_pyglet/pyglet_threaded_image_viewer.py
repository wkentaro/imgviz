import threading

import pyglet

from .pyglet_imshow import _ndarray_to_imagedata


class PygletThreadedImageViewer(pyglet.window.Window):

    def __init__(self, play=True, interval=0.5):
        self._play = play
        self._next = False

        self._updated_at = 0
        self._interval = interval

        self.sprite = None

        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._init_and_start_app)
        self.thread.daemon = True  # terminate when main thread exit
        self.thread.start()

        pyglet.clock.schedule_interval(self.on_update, 1 / 100)

    def _init_and_start_app(self):
        super(PygletThreadedImageViewer, self).__init__()
        pyglet.app.run()

    def imshow(self, image):
        imagedata = _ndarray_to_imagedata(image)
        with self.lock:
            if self.sprite is None:
                self.sprite = pyglet.sprite.Sprite(imagedata)
            else:
                self.sprite.image = imagedata
            inserted_at = self._updated_at

        while True:
            with self.lock:
                if self._play and \
                        self._updated_at > (inserted_at + self._interval):
                    break
                elif self._next:
                    self._next = False
                    break

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

    def on_key_press(self, symbol, modifiers):

        def usage():
            print('Usage: ')
            print('  h: show help')
            print('  q: close window')
            print('  n: next image')
            print('  s: toggle play')

        def short_usage():
            print("Press 'h' to show help")

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
        else:
            short_usage()
