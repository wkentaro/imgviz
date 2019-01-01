import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.arc2017()

    rgb = data['rgb']

    aabb1 = np.array([100, 0], dtype=int)
    aabb2 = np.array([480, 640], dtype=int)
    viz = imgviz.draw.rectangle(rgb, aabb1, aabb2, color=(0, 255, 0), width=10)

    text = 'bin'
    size = 50
    height, width = imgviz.draw.text_size(text=text, size=size)
    viz = imgviz.draw.rectangle(
        viz,
        aabb1,
        aabb1 + [height, width],
        color=(0, 255, 0),
        fill=(0, 255, 0),
    )
    viz = imgviz.draw.text(
        viz, yx=aabb1, text=text, color=(0, 0, 0), size=size
    )

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(121)
    plt.title('original')
    plt.imshow(rgb)
    plt.axis('off')

    plt.subplot(122)
    plt.title('aabb1: {}\naabb2: {}'.format(aabb1, aabb2))
    plt.imshow(viz)
    plt.axis('off')

    out_file = osp.join(here, '.readme/draw.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    img = imgviz.io.imread(out_file)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
