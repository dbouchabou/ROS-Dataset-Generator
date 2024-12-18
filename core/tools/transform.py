import numpy as np
import tf.transformations


def camera_frame_to_image(points, K):
    """Compute points coordinates in the image frame from their coordinates in
    the camera frame
    Args:
        points (ndarray (N, 3)): a set of points
        K (ndarray (3, 3)): the internal calibration matrix
    Returns:
        ndarray (N, 2): points image coordinates
    """
    # Project the points onto the image plan, the obtained coordinates are
    # defined up to a scaling factor
    points_projection = np.dot(points, np.transpose(K))

    # Get the points' coordinates in the image frame dividing by the third
    # coordinate
    points_image = points_projection[:, :2] / points_projection[:, 2][:, np.newaxis]

    return points_image


def apply_rigid_motion(points, HTM, rot_only=False):
    """Give points' coordinates in a new frame obtained after rotating (R)
    and translating (T) the current one
    Args:
        points (ndarray (N, 3)): a set of points
        HTM (ndarray (4, 4)): a homogeneous transform matrix
    Returns:
        ndarray (N, 3): points new coordinates
    """
    # Number of points we want to move
    nb_points = np.shape(points)[0]

    # Use homogenous coordinates
    homogeneous_points = np.ones((nb_points, 4))
    if rot_only:
        homogeneous_points = np.zeros((nb_points, 4))

    homogeneous_points[:, :-1] = points

    # Compute points coordinates after the rigid motion
    points_new = np.dot(homogeneous_points, np.transpose(HTM[:3, :]))

    return points_new


def pose_to_transform_matrix(pose):
    """
    Convert a pose dictionary to a transformation matrix.

    Args:
        pose (dict): Pose dictionary containing position and orientation.

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    position = pose["position"]
    orientation = pose["orientation"]

    T = np.array([position["x"], position["y"], position["z"]])
    q = np.array(
        [orientation["x"], orientation["y"], orientation["z"], orientation["w"]]
    )

    HTM = tf.transformations.quaternion_matrix(q)
    HTM[0:3, 3] = T

    return HTM


def inverse_transform_matrix(HTM):
    """
    Compute the inverse of a homogeneous transformation matrix.

    Args:
        HTM (np.ndarray): 4x4 homogeneous transformation matrix.

    Returns:
        np.ndarray: Inverse of the input matrix.
    """
    R = HTM[:3, :3]
    t = HTM[:3, 3]

    HTM_inverse = np.zeros_like(HTM)
    HTM_inverse[:3, :3] = np.transpose(R)
    HTM_inverse[:3, 3] = -np.dot(np.transpose(R), t)
    HTM_inverse[3, 3] = 1

    return HTM_inverse
