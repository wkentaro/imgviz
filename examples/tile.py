import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.arc2017()

    rgb = data['rgb']

    bboxes = data['bboxes'].astype(int)
    crops = [data['rgb'][b[0]:b[2], b[1]:b[3]] for b in bboxes]
    tiled = imgviz.tile(imgs=crops, border=(255, 255, 255))

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(121)
    plt.imshow(rgb)
    plt.axis('off')

    plt.subplot(122)
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
