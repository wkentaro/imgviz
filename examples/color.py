import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.arc2017()

    rgb = data['rgb']
    gray = imgviz.rgb2gray(rgb)
    rgb2 = imgviz.gray2rgb(gray)

    # -------------------------------------------------------------------------

    fig = plt.figure(dpi=150)

    plt.subplot(131)
    plt.title('original')
    plt.imshow(rgb)
    plt.axis('off')

    plt.subplot(132)
    plt.title('rgb2gray:\n{}'.format(gray.shape))
    plt.imshow(gray)
    plt.axis('off')

    plt.subplot(133)
    plt.title('rgb2gray, gray2rgb:\n{}'.format(rgb2.shape))
    plt.imshow(rgb2)
    plt.axis('off')

    out_file = osp.join(here, '.readme/color.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    plt.imshow(plt.imread(out_file))
    plt.axis('off')
    plt.show()
