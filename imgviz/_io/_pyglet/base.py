try:
    import pyglet
except ImportError:
    pyglet = None


def check_pyglet_available():
    if pyglet is None:
        raise ImportError(
            "pyglet is not installed, run following: pip install pyglet"
        )
    return pyglet
