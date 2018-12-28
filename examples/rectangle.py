import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.arc2017()

    rgb = data['rgb']

    aabb1 = (100, 0)
    aabb2 = (480, 640)
    viz = imgviz.rectangle(rgb, aabb1, aabb2, color=(0, 255, 0), width=10)

    # -------------------------------------------------------------------------

    plt.figure(dpi=150)

    plt.subplot(121)
    plt.title('original')
    plt.imshow(rgb)
    plt.axis('off')

    plt.subplot(122)
    plt.title('aabb1: {}\naabb2: {}'.format(aabb1, aabb2))
    plt.imshow(viz)
    plt.axis('off')

    out_file = osp.join(here, '.readme/rectangle.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    plt.imshow(plt.imread(out_file))
    plt.axis('off')
    plt.show()
