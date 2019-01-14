import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.kitti_odometry()

    img = imgviz.plot_trajectory(data['transforms'])

    # -------------------------------------------------------------------------

    out_file = osp.join(here, '.readme/trajectory.jpg')
    imgviz.io.imsave(out_file, img)

    img = imgviz.io.imread(out_file)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
