import matplotlib.pyplot as plt
import numpy as np

import imgviz


if __name__ == '__main__':
    data_file = 'data/arc2017/1532900700692405455/data.npz'
    data = np.load(data_file)

    rgb = data['img']
    gray = imgviz.rgb2gray(rgb)
    depth = imgviz.depth2rgb(data['depth'], min_value=0.3, max_value=1)

    tiled = imgviz.tile(
        imgs=[rgb, gray, depth],
        shape=(2, 2),
        cval=(255, 255, 255),
        border=[None, (0, 255, 0), None],
    )

    plt.imshow(tiled)
    plt.show()
