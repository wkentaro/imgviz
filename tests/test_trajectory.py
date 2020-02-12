import numpy as np

import imgviz


def test_trajectory():
    data = imgviz.data.kitti_odometry()

    img = imgviz.trajectory.plot_trajectory(data["transforms"])
    assert isinstance(img, np.ndarray)
