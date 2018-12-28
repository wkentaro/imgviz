import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.arc2017()

    # compose masks to class label image
    label = np.full(data['rgb'].shape[:2], 0, dtype=np.int32)
    for l, mask in zip(data['labels'], data['masks']):
        label[mask == 1] = l

    labelviz = imgviz.label2rgb(label)

    label_names = [
        '{}:{}'.format(i, n) for i, n in enumerate(data['class_names'])
    ]
    labelviz_withname = imgviz.label2rgb(label, label_names=label_names)

    # -------------------------------------------------------------------------

    fig = plt.figure(dpi=150)

    plt.subplot(131)
    plt.title('rgb')
    plt.imshow(data['rgb'])
    plt.axis('off')

    plt.subplot(132)
    plt.title('label\n(colorized)')
    plt.imshow(labelviz)
    plt.axis('off')

    plt.subplot(133)
    plt.title('label\n(colorized + names)')
    plt.imshow(labelviz_withname)
    plt.axis('off')

    out_file = osp.join(here, '.readme/label2rgb.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    plt.imshow(plt.imread(out_file))
    plt.axis('off')
    plt.show()
