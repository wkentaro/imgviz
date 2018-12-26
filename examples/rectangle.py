import matplotlib.pyplot as plt
import numpy as np

import imgviz


if __name__ == '__main__':
    data_file = 'data/arc2017/1532900700692405455/data.npz'
    data = np.load(data_file)

    rgb = data['img']

    aabb1 = (100, 0)
    aabb2 = (480, 640)
    viz = imgviz.rectangle(rgb, aabb1, aabb2, color=(0, 255, 0), width=10)

    plt.subplot(121)
    plt.title('original')
    plt.imshow(rgb)
    plt.subplot(122)
    plt.title('{}, {}'.format(aabb1, aabb2))
    plt.imshow(viz)
    plt.show()
