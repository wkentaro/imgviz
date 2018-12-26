import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data_file = osp.join(here, 'data/arc2017/1532900700692405455/data.npz')
    data = np.load(data_file)

    rgb = data['img']

    H, W = rgb.shape[:2]
    H2, W2 = int(round(H * 0.1)), int(round(W * 0.1))
    rgb_resized = imgviz.resize(rgb, height=H2, width=W2)

    plt.figure(dpi=150)
    plt.subplot(121)
    plt.title('{}'.format(rgb.shape))
    plt.imshow(rgb)
    plt.subplot(122)
    plt.title('{}'.format(rgb_resized.shape))
    plt.imshow(rgb_resized)

    out_file = osp.join(here, '.readme/resize.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    plt.imshow(plt.imread(out_file))
    plt.axis('off')
    plt.show()
