import matplotlib.pyplot as plt
import numpy as np

import imgviz


def test_pyplot():
    backend = plt.get_backend()
    plt.switch_backend('agg')

    fig = plt.figure()
    x = y = [0, 1, 2]
    plt.plot(x, y)

    plt.switch_backend(backend)

    img = imgviz.io.pyplot_fig2arr(fig)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
