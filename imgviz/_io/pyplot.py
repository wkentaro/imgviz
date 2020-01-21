import io

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image


def pyplot_to_numpy():
    f = io.BytesIO()
    plt.savefig(
        f,
        bbox_inches='tight',
        transparent='True',
        pad_inches=0,
        format='jpeg',
    )
    plt.close()
    f.seek(0)
    return np.asarray(PIL.Image.open(f))
