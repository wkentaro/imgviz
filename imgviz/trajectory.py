from ._io import pyplot_to_numpy
from .external import transformations as tf


def plot_trajectory(
    transforms,
    is_relative=False,
    mode="xz",
    style="b.",
    axis=True,
):
    """Plot the trajectory using transform matrices

    Parameters
    ----------
    transforms: numpy.ndarray
        transform matrices with the shape of [N, 4, 4]
        where N is the # of poses.
    is_relative: bool
        True for relative poses. default: False.
    mode: str
        x and y axis of trajectory. default: 'xz' following kitti format.
    style: str
        style of ploting, default: 'b.'
    axis: bool
        False to disable axis.

    Returns
    -------
    dst: numpy.ndarray
        trajectory

    """
    import matplotlib.pyplot as plt  # slow

    if is_relative:
        for i in range(1, len(transforms)):
            transforms[i] = transforms[i - 1].dot(transforms[i])

    if len(mode) != 2 and all(x in "xyz" for x in mode):
        raise ValueError("Unsupported mode: {}".format(mode))

    x = []
    y = []
    index_x = "xyz".index(mode[0])
    index_y = "xyz".index(mode[1])
    for T in transforms:
        translate = tf.translation_from_matrix(T)
        x.append(translate[index_x])
        y.append(translate[index_y])

    # swith backend to agg for supporting no display mode
    backend = plt.get_backend()
    plt.switch_backend("agg")

    plt.plot(x, y, style)

    if not axis:
        plt.axis("off")

    dst = pyplot_to_numpy()
    plt.close()

    # switch back backend
    plt.switch_backend(backend)

    return dst
