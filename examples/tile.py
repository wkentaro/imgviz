import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.arc2017()

    rgb = data['rgb']

    crops = []
    for bbox, mask in zip(data['bboxes'], data['masks']):
        y1, x1, y2, x2 = bbox.astype(int)
        rgb_crop = rgb[y1:y2, x1:x2].copy()
        mask_crop = mask[y1:y2, x1:x2]
        rgb_crop[mask_crop != 1] = 0
        crops.append(rgb_crop)

    tiled = imgviz.tile(
        imgs=crops,
        border=(255, 255, 255),
    )

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
