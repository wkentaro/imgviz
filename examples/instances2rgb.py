import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.arc2017()

    captions = [data['class_names'][l] for l in data['labels']]
    insviz = imgviz.instances2rgb(
        src=data['rgb'],
        bboxes=data['bboxes'],
        labels=data['labels'],
        captions=captions,
    )

    # -------------------------------------------------------------------------

    fig = plt.figure(dpi=200)

    plt.subplot(121)
    plt.title('rgb')
    plt.imshow(data['rgb'])
    plt.axis('off')

    plt.subplot(122)
    plt.title('instances\n(colorized)')
    plt.imshow(insviz)
    plt.axis('off')

    out_file = osp.join(here, '.readme/instances2rgb.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    img = imgviz.io.imread(out_file)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
