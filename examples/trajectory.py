import os.path as osp

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    data = imgviz.data.kitti()

    img = imgviz.plot_trajectory(data['transforms'])
    out_file = osp.join(here, '.readme/trajectory.jpg')
    imgviz.io.imsave(out_file, img)

    img = imgviz.io.imread(out_file)
    imgviz.io.pyglet_imshow(img)
    imgviz.io.pyglet_run()
