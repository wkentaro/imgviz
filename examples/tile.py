import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data_file = osp.join(here, 'data/arc2017/1532900700692405455/data.npz')
    data = np.load(data_file)

    rgb = data['img']
    gray = imgviz.rgb2gray(rgb)
    rgbT = rgb.transpose(1, 0, 2)

    tiled = imgviz.tile(
        imgs=[rgb, gray, rgbT],
        shape=(1, 4),
        border=(255, 255, 255),
    )

    plt.figure(dpi=150)
    plt.imshow(tiled)
    plt.axis('off')

    out_file = osp.join(here, '.readme/tile.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    plt.imshow(plt.imread(out_file))
    plt.axis('off')
    plt.show()
