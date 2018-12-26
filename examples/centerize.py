import matplotlib.pyplot as plt
import numpy as np

import imgviz


if __name__ == '__main__':
    data_file = 'data/arc2017/1532900700692405455/data.npz'
    data = np.load(data_file)

    rgb = data['img']

    H, W = rgb.shape[:2]
    centerized1 = imgviz.centerize(rgb, shape=(H, H))

    rgb = rgb.transpose(1, 0, 2)
    centerized2 = imgviz.centerize(rgb, shape=(H, H))

    plt.subplot(121)
    plt.title('{}'.format(centerized1.shape))
    plt.imshow(centerized1)
    plt.subplot(122)
    plt.title('{}'.format(centerized2.shape))
    plt.imshow(centerized2)
    plt.show()
