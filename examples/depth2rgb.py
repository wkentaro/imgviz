import matplotlib.pyplot as plt
import numpy as np

import imgviz


if __name__ == '__main__':
    data_file = 'data/arc2017/1532900700692405455/data.npz'
    data = np.load(data_file)

    depth = data['depth']

    color = imgviz.depth2rgb(depth, min_value=0.3, max_value=1)
    plt.imshow(color)
    plt.show()
