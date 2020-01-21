import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.voc()

    rgb = data['rgb']
    label = data['class_label']

    labelviz = imgviz.label2rgb(label)

    label_names = [
        '{}:{}'.format(i, n) for i, n in enumerate(data['class_names'])
    ]
    labelviz_withname1 = imgviz.label2rgb(
        label, label_names=label_names, font_size=25
    )
    labelviz_withname2 = imgviz.label2rgb(
        label, label_names=label_names, font_size=25, loc='lt'
    )
    img = imgviz.color.rgb2gray(rgb)
    labelviz_withimg = imgviz.label2rgb(label, img=img)

    # -------------------------------------------------------------------------

    fig = plt.figure(dpi=200)

    plt.subplot(131)
    plt.title('+img')
    plt.imshow(labelviz_withimg)
    plt.axis('off')

    plt.subplot(132)
    plt.title('loc=centroid')
    plt.imshow(labelviz_withname1)
    plt.axis('off')

    plt.subplot(133)
    plt.title('loc=lt')
    plt.imshow(labelviz_withname2)
    plt.axis('off')

    out_file = osp.join(here, '.readme/label2rgb.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    img = imgviz.io.imread(out_file)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
