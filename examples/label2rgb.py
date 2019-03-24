import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.arc2017()

    labelviz = imgviz.label2rgb(data['class_label'])

    label_names = [
        '{}:{}'.format(i, n) for i, n in enumerate(data['class_names'])
    ]
    labelviz_withname = imgviz.label2rgb(
        data['class_label'], label_names=label_names
    )
    img = imgviz.color.rgb2gray(data['rgb'])
    labelviz_withimg = imgviz.label2rgb(data['class_label'], img=img)

    # -------------------------------------------------------------------------

    fig = plt.figure(dpi=200)

    plt.subplot(141)
    plt.title('rgb')
    plt.imshow(data['rgb'])
    plt.axis('off')

    plt.subplot(142)
    plt.title('label')
    plt.imshow(labelviz)
    plt.axis('off')

    plt.subplot(143)
    plt.title('label\n(+names)')
    plt.imshow(labelviz_withname)
    plt.axis('off')

    plt.subplot(144)
    plt.title('label\n(+img)')
    plt.imshow(labelviz_withimg)
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
