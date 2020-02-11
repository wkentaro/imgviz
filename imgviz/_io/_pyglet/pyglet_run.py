from .base import check_pyglet_available


def pyglet_run():
    # type: () -> None
    """Start pyglet mainloop."""
    pyglet = check_pyglet_available()

    return pyglet.app.run()
