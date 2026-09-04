import numpy as np

import imgviz


def test_kitti_odometry_parses_pose_values() -> None:
    data = imgviz.data.kitti_odometry()
    assert isinstance(data, dict)

    transforms = data["transforms"]
    assert len(transforms) == 4541
    assert transforms[0].shape == (4, 4)
    assert transforms[0].dtype == np.float64

    # Each line's 12 floats fill the top three rows row-major, and the bottom
    # row is the appended homogeneous [0, 0, 0, 1]; the translation column is
    # the 4th value of each triplet.
    np.testing.assert_array_equal(transforms[1][3], [0, 0, 0, 1])
    np.testing.assert_array_equal(
        transforms[1][:3, 3], [-4.690294e-02, -2.839928e-02, 8.586941e-01]
    )
