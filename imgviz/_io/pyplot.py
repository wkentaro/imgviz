import io

import numpy as np
import PIL.Image


def pyplot_to_numpy():
    """Convert pyplot state to numpy array.

    Parameters
    ----------

    Returns
    -------
    arr: numpy.ndarray
        Plotted image.

    """
    import matplotlib.pyplot as plt  # slow

    f = io.BytesIO()
    plt.savefig(
        f,
        bbox_inches="tight",
        transparent="True",
        pad_inches=0,
        format="jpeg",
    )
    plt.close()
    f.seek(0)
    arr = np.asarray(PIL.Image.open(f))
    return arr


def pyplot_imshow(image):
    import matplotlib.pyplot as plt  # slow

    plt.cla()
    plt.imshow(image)
    plt.tight_layout()
    plt.show()
