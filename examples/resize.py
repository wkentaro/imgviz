import matplotlib.pyplot as plt
import numpy as np

import imgviz


if __name__ == '__main__':
    data_file = 'data/arc2017/1532900700692405455/data.npz'
    data = np.load(data_file)

    rgb = data['img']

    H, W = rgb.shape[:2]
    H2, W2 = int(round(H * 0.1)), int(round(W * 0.1))
    rgb_resized = imgviz.resize(rgb, height=H2, width=W2)

    plt.subplot(121)
    plt.title('{}'.format(rgb.shape))
    plt.imshow(rgb)
    plt.subplot(122)
    plt.title('{}'.format(rgb_resized.shape))
    plt.imshow(rgb_resized)
    plt.show()
