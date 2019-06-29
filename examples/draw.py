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
    viz = imgviz.draw.rectangle(
        rgb, aabb1, aabb2, outline=(0, 255, 0), width=10
    )
    viz = viz.copy()  # to make it writable

    y1, x1 = aabb1
    y2, x2 = aabb2
    viz[y1:y2, x1:x2] = imgviz.draw.text_in_rectangle(
        viz[y1:y2, x1:x2],
        loc='lt',
        text='bin',
        size=50,
        background=(0, 255, 0),
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
