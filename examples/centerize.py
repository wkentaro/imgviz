import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.arc2017()

    rgb = data['rgb']

    H, W = rgb.shape[:2]
    centerized1 = imgviz.centerize(rgb, shape=(H, H))

    rgb_T = rgb.transpose(1, 0, 2)
    centerized2 = imgviz.centerize(rgb_T, shape=(H, H))

    # -------------------------------------------------------------------------

    fig = plt.figure(dpi=200)

    plt.subplot(131)
    plt.title('original')
    plt.axis('off')
    plt.imshow(rgb)

    plt.subplot(132)
    plt.title('centerized1:\n{}'.format(centerized1.shape))
    plt.imshow(centerized1)
    plt.axis('off')

    plt.subplot(133)
    plt.title('centerized2:\n{}'.format(centerized2.shape))
    plt.imshow(centerized2)
    plt.axis('off')

    out_file = osp.join(here, '.readme/centerize.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    img = imgviz.io.imread(out_file)
    imgviz.io.pyglet_imshow(img)
    imgviz.io.pyglet_run()
