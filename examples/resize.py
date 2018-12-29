import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.arc2017()

    rgb = data['rgb']

    H, W = rgb.shape[:2]
    rgb_resized = imgviz.resize(rgb, height=0.1)

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(121)
    plt.title('rgb:\n{}'.format(rgb.shape))
    plt.imshow(rgb)
    plt.axis('off')

    plt.subplot(122)
    plt.title('rgb_resized:\n{}'.format(rgb_resized.shape))
    plt.imshow(rgb_resized)
    plt.axis('off')

    out_file = osp.join(here, '.readme/resize.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    img = imgviz.io.imread(out_file)
    imgviz.io.pyglet_imshow(img)
    imgviz.io.pyglet_run()
