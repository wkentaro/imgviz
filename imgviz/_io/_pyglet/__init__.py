# flake8: noqa

from .base import check_pyglet_available

from .pyglet_imshow import pyglet_imshow

from .pyglet_run import pyglet_run

try:
    from .pyglet_threaded_image_viewer import PygletThreadedImageViewer
except ImportError:

    class PygletThreadedImageViewer(object):  # type: ignore

        def __init__(self, *args, **kwargs):
            check_pyglet_available()
