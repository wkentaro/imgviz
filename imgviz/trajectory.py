import numpy as np

# matplotlib.use('agg')
import matplotlib.pyplot as plt


def pose_to_transform(pose):
    theta = pose[:3]
    t = pose[3:6]
    R = eulerAnglesToRotationMatrix(theta)
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1

    return T


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)

    epsilon = 1e-2
    if n > epsilon:
        print(n)
    return n < epsilon


def R_to_angle(Rt):
    # Ground truth pose is present as [R | t]
    # R: Rotation Matrix, t: translation vector
    # transform matrix to angles
    # Rt = np.reshape(np.array(Rt), (3, 4))
    assert Rt.shape == (3, 4)
    t = Rt[:, -1]
    R = Rt[:, :3]

    assert(isRotationMatrix(R))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    theta = [x, y, z]
    if abs(x) > np.pi or abs(y) > np.pi or abs(z) > np.pi:
        print(theta)

    pose_6 = np.concatenate((theta, t))
    assert(pose_6.shape == (6,))
    return pose_6


def fig2array(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    w, h = fig.canvas.get_width_height()
    data = data.reshape((h, w, 3))

    return data


def plot_trajectory_with_pose(poses, is_relative=True, style='b.'):
    """Plot the trajectory using 6DOF poses 

    Parameters
    ----------
    poses: numpy.ndarray
        6DOF poses with the shape of [N, 6] where N is the # of poses.
    is_relative: bool
        True for relative poses.
    style: str
        style of ploting, default: 'b.'
    dst: numpy.ndarray
        trajectory
    """
    if is_relative:
        T = [pose_to_transform(poses[i]) for i in range(poses.shape[0])]

        for i in range(1, len(T)):
            T[i] = T[i - 1].dot(T[i])

        poses = np.array([R_to_angle(t[:3]) for t in T])

    x = poses[:, 3]
    z = poses[:, 5]

    fig = plt.figure()
    plt.axis('off')
    plt.plot(x, z, style)

    img = fig2array(fig)
    plt.close()

    return img


def plot_trajectory_with_transform(T, is_relative=True, style='b.'):
    """Plot the trajectory using transform matrices

    Parameters
    ----------
    T: numpy.ndarray
        transform matrices with the shape of [N, 4, 4] 
        where N is the # of poses.
    is_relative: bool
        True for relative poses.
    style: str
        style of ploting, default: 'b.'
    dst: numpy.ndarray
        trajectory
    """
    if is_relative:
        for i in range(1, len(T)):
            T[i] = T[i - 1].dot(T[i])

    poses = np.array([R_to_angle(t[:3]) for t in T])

    x = poses[:, 3]
    z = poses[:, 5]

    fig = plt.figure()
    plt.axis('off')
    plt.plot(x, z, style)

    img = fig2array(fig)
    plt.close()

    return img
