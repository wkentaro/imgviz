import numpy as np


def pyplot_fig2arr(fig):
    fig.canvas.draw()
    arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    arr = arr.reshape((height, width, 3))
    return arr
