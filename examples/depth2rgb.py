import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.arc2017()

    depthviz = imgviz.depth2rgb(data['depth'], min_value=0.3, max_value=1)

    # -------------------------------------------------------------------------

    fig = plt.figure(dpi=200)

    plt.subplot(121)
    plt.title('rgb')
    plt.imshow(data['rgb'])
    plt.axis('off')

    plt.subplot(122)
    plt.title('depth (colorized)')
    plt.imshow(depthviz)
    plt.axis('off')

    out_file = osp.join(here, '.readme/depth2rgb.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    plt.imshow(plt.imread(out_file))
    plt.axis('off')
    plt.show()
