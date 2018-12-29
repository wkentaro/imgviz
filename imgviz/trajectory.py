import matplotlib.pyplot as plt
import transformations as tf

from ._io import pyplot_fig2arr


def plot_trajectory(transforms, is_relative=False, style='b.', mode='xz'):
    """Plot the trajectory using transform matrices

    Parameters
    ----------
    transforms: numpy.ndarray
        transform matrices with the shape of [N, 4, 4]
        where N is the # of poses.
    is_relative: bool
        True for relative poses. default: False.
    style: str
        style of ploting, default: 'b.'
    mode: str
        x and y axis of trajectory. default: 'xz' following kitti format.

    Returns
    -------
    dst: numpy.ndarray
        trajectory
    """
    if is_relative:
        for i in range(1, len(transforms)):
            transforms[i] = transforms[i - 1].dot(transforms[i])

    if len(mode) != 2 and all(x in 'xyz' for x in mode):
        raise ValueError('Unsupported mode: {}'.format(mode))

    x = []
    y = []
    index_x = 'xyz'.index(mode[0])
    index_y = 'xyz'.index(mode[1])
    for T in transforms:
        translate = tf.translation_from_matrix(T)
        x.append(translate[index_x])
        y.append(translate[index_y])

    fig = plt.figure()
    plt.axis('off')
    plt.plot(x, y, style)

    dst = pyplot_fig2arr(fig)
    plt.close()

    return dst
