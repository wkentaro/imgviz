import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.kitti()
    img, flow = data['flow']
    flow_rgb = imgviz.flow2rgb(flow)

    # -------------------------------------------------------------------------

    fig = plt.figure(dpi=200)
    plt.subplot(1, 2, 1)
    plt.title('image')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('flow')
    plt.imshow(flow_rgb)
    plt.axis('off')

    out_file = osp.join(here, '.readme/flow.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    img = imgviz.io.imread(out_file)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
