import matplotlib.pyplot as plt
import numpy as np

import imgviz


if __name__ == '__main__':
    data_file = 'data/arc2017/1532900700692405455/data.npz'
    data = np.load(data_file)

    rgb = data['img']
    gray = imgviz.rgb2gray(rgb)
    rgb2 = imgviz.gray2rgb(gray)

    plt.subplot(131)
    plt.title('{}, {}'.format(rgb.shape, rgb.dtype))
    plt.imshow(rgb)
    plt.subplot(132)
    plt.title('{}, {}'.format(gray.shape, gray.dtype))
    plt.imshow(gray)
    plt.subplot(133)
    plt.title('{}, {}'.format(rgb2.shape, rgb2.dtype))
    plt.imshow(rgb2)
    plt.show()
