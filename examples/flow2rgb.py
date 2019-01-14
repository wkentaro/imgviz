import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.middlebury()

    rgb = data['rgb']
    flowviz = imgviz.flow2rgb(data['flow'])

    # -------------------------------------------------------------------------

    fig = plt.figure(dpi=200)
    plt.subplot(121)
    plt.title('image')
    plt.imshow(rgb)
    plt.axis('off')

    plt.subplot(122)
    plt.title('flow')
    plt.imshow(flowviz)
    plt.axis('off')

    out_file = osp.join(here, '.readme/flow2rgb.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    img = imgviz.io.imread(out_file)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
