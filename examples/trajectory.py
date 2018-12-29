import matplotlib.pyplot as plt
import numpy as np

import imgviz
from imgviz.trajectory import R_to_angle


def read_poses(root):
    with open(root, 'r') as posefile:
        # read ground truth
        poses = []
        rel_poses = []
        T = []
        rel_T = []
        for one_line in posefile:
            one_line = one_line.split(' ')
            Rt = [float(pose) for pose in one_line] + [0, 0, 0, 1]

            Rt = np.reshape(np.array(Rt), (4, 4))
            assert abs(Rt[3].sum() - 1) < 1e-5
            T.append(Rt)

        for i in range(len(T) - 1):
            T_i = np.linalg.inv(T[i]).dot(T[i + 1])
            assert abs(T_i[3].sum() - 1) < 1e-5
            rel_T.append(T_i)

        for i in range(len(T)):
            gt = R_to_angle(T[i][:3, :])
            poses.append(gt)

        for i in range(len(rel_T)):
            gt = R_to_angle(rel_T[i][:3, :])
            rel_poses.append(gt)

    T = np.array(T, dtype=np.float32)
    rel_T = np.array(rel_T, dtype=np.float32)
    poses = np.array(poses, dtype=np.float32)
    rel_poses = np.array(rel_poses, dtype=np.float32)

    return T, rel_T, poses, rel_poses


if __name__ == '__main__':
    file = 'examples/data/KITTI_VO_pose/00.txt'
    T, rel_T, poses, rel_poses = read_poses(file)

    img = imgviz.plot_trajectory_with_pose(poses, False, style='b.')
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    plt.imsave('trajectory0.png', img)

    img = imgviz.plot_trajectory_with_pose(rel_poses, is_relative=True)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    # plt.imsave('trajectory1.png', img)

    img = imgviz.plot_trajectory_with_transform(T, is_relative=False)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    # plt.imsave('trajectory1.png', img)

    img = imgviz.plot_trajectory_with_transform(rel_T, is_relative=True)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    # plt.imsave('trajectory1.png', img)
