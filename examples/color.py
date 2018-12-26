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
    rgb2 = imgviz.gray2rgb(gray)

    fig = plt.figure(dpi=150)
    plt.subplot(131)
    plt.title('{}, {}'.format(rgb.shape, rgb.dtype))
    plt.imshow(rgb)
    plt.axis('off')
    plt.subplot(132)
    plt.title('{}, {}'.format(gray.shape, gray.dtype))
    plt.imshow(gray)
    plt.axis('off')
    plt.subplot(133)
    plt.title('{}, {}'.format(rgb2.shape, rgb2.dtype))
    plt.imshow(rgb2)
    plt.axis('off')

    out_file = osp.join(here, '.readme/color.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    plt.imshow(plt.imread(out_file))
    plt.axis('off')
    plt.show()
