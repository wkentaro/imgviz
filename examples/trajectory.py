import os.path as osp

import numpy as np

import imgviz


here = osp.dirname(osp.abspath(__file__))


def read_poses(filename):
    with open(filename, 'r') as f:
        transforms = []
        for one_line in f:
            one_line = one_line.split(' ')
            Rt = [float(pose) for pose in one_line] + [0, 0, 0, 1]

            Rt = np.reshape(np.array(Rt), (4, 4))
            assert abs(Rt[3].sum() - 1) < 1e-5
            transforms.append(Rt)
    return transforms


if __name__ == '__main__':
    file = osp.join(here, 'data/KITTI_VO_pose/00.txt')
    transforms = read_poses(file)

    img = imgviz.plot_trajectory(transforms, is_relative=False)
    out_file = osp.join(here, '.readme/trajectory.jpg')
    imgviz.io.imsave(out_file, img)

    img = imgviz.io.imread(out_file)
    imgviz.io.pyglet_imshow(img)
    imgviz.io.pyglet_run()
