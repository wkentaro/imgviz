import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.arc2017()

    nchannel_viz = imgviz.nchannel2rgb(data['res4'], shape=data['rgb'].shape)

    # -------------------------------------------------------------------------

    fig = plt.figure(dpi=200)

    plt.subplot(121)
    plt.title('rgb')
    plt.imshow(data['rgb'])
    plt.axis('off')

    plt.subplot(122)
    plt.title('res4 (colorized)')
    plt.imshow(nchannel_viz)
    plt.axis('off')

    out_file = osp.join(here, '.readme/nchannel2rgb.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    img = imgviz.io.imread(out_file)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
