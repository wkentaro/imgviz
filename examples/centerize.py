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
    centerized1 = imgviz.centerize(rgb, shape=(H, H))

    rgb = rgb.transpose(1, 0, 2)
    centerized2 = imgviz.centerize(rgb, shape=(H, H))

    fig = plt.figure(dpi=150)
    plt.subplot(121)
    plt.title('{}'.format(centerized1.shape))
    plt.imshow(centerized1)
    plt.axis('off')
    plt.subplot(122)
    plt.title('{}'.format(centerized2.shape))
    plt.imshow(centerized2)
    plt.axis('off')

    out_file = osp.join(here, '.readme/centerize.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    plt.imshow(plt.imread(out_file))
    plt.axis('off')
    plt.show()
