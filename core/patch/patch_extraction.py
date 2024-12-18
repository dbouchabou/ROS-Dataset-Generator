import numpy as np
import cv2
from typing import List, Dict, Tuple, Any

from core.tools.robots import calculate_wheel_trajectories, robot_to_image
from core.tools.transform import (
    pose_to_transform_matrix,
    inverse_transform_matrix,
    apply_rigid_motion,
)
from core.tools.data_processing import filter_points_in_square, select_points
from core.data_models import DatasetConfig


def extract_rectangular_patches(
    image_data: Dict[str, Any],
    odometry_data: List[Dict[str, Any]],
    config: DatasetConfig,
    D: float = 1.0,
    d: float = 0.1,
    patch_width: float = 0.65,
    patch_height: float = 0.5,
    overlap: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Extract rectangular patches along the robot trajectory with controlled overlap.

    Args:
        image_data (Dict[str, Any]): Image data dictionary.
        odometry_data (List[Dict[str, Any]]): List of odometry data dictionaries.
        config (DatasetConfig): Dataset configuration object.
        D (float): Distance parameter for filtering points in square.
        d (float): Distance threshold for selecting points.
        patch_width (float): Width of the patch in meters.
        patch_height (float): Height of the patch in meters.
        overlap (float): Fraction of overlap between consecutive patches (0 to 1).

    Returns:
        List[Dict[str, Any]]: List of extracted patches with metadata.
    """
    image = image_data["image"]
    timestamp = image_data["timestamp"]

    # Find the closest odometry data to the image timestamp
    closest_odom = min(odometry_data, key=lambda x: abs(x["timestamp"] - timestamp))

    # Calculate transformations
    ROBOT_TO_WORLD = pose_to_transform_matrix(closest_odom)
    WORLD_TO_ROBOT = inverse_transform_matrix(ROBOT_TO_WORLD)
    ROBOT_TO_CAM = inverse_transform_matrix(config.cam_to_robot)

    # Transform odometry data to robot frame
    odometry_data_robot = [
        {
            **odom,
            "position": apply_rigid_motion(
                np.array(
                    [
                        odom["position"]["x"],
                        odom["position"]["y"],
                        odom["position"]["z"],
                    ]
                ),
                WORLD_TO_ROBOT,
            )[0],
        }
        for odom in odometry_data
    ]

    # Filter and select points
    filtered_odometry_data = filter_points_in_square(odometry_data_robot, D)
    selected_points = select_points(filtered_odometry_data, d)

    # Calculate wheel trajectories
    left_traj, right_traj = calculate_wheel_trajectories(
        selected_points, wheel_distance=patch_width
    )

    # Transform trajectories to image coordinates
    center_traj_image = robot_to_image(selected_points, ROBOT_TO_CAM, config)
    left_traj_image = robot_to_image(left_traj, ROBOT_TO_CAM, config)
    right_traj_image = robot_to_image(right_traj, ROBOT_TO_CAM, config)

    patches = []
    last_patch_center = None
    min_distance = patch_height * (1 - overlap)

    for i in range(len(center_traj_image)):
        center_point = np.array(
            [
                center_traj_image[i]["position"]["x"],
                center_traj_image[i]["position"]["y"],
            ]
        )

        # Check if we're far enough from the last patch
        if last_patch_center is not None:
            distance = np.linalg.norm(center_point - last_patch_center)
            if distance < min_distance:
                continue

        left_point = np.array(
            [left_traj_image[i]["position"]["x"], left_traj_image[i]["position"]["y"]]
        )
        right_point = np.array(
            [right_traj_image[i]["position"]["x"], right_traj_image[i]["position"]["y"]]
        )

        # Calculate patch corners
        direction = right_point - left_point
        perpendicular = np.array([-direction[1], direction[0]])
        perpendicular = (
            perpendicular / np.linalg.norm(perpendicular) * (patch_height / 2)
        )

        top_left = left_point - perpendicular
        top_right = right_point - perpendicular
        bottom_left = left_point + perpendicular
        bottom_right = right_point + perpendicular

        # Extract patch
        src_pts = np.array(
            [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
        )
        dst_pts = np.array(
            [[0, 0], [patch_width, 0], [patch_width, patch_height], [0, patch_height]],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        patch = cv2.warpPerspective(
            image, M, (int(patch_width * 100), int(patch_height * 100))
        )

        patches.append(
            {
                "patch": patch,
                "timestamp": selected_points[i]["timestamp"],
                "position": selected_points[i]["position"],
                "corners": {
                    "top_left": top_left.tolist(),
                    "top_right": top_right.tolist(),
                    "bottom_left": bottom_left.tolist(),
                    "bottom_right": bottom_right.tolist(),
                },
            }
        )

        last_patch_center = center_point

    return patches


def save_patches(patches: List[Dict[str, Any]], output_dir: str):
    """Save extracted patches and their metadata."""
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)

    for i, patch_data in enumerate(patches):
        patch = patch_data["patch"]
        metadata = {
            "timestamp": patch_data["timestamp"],
            "position": patch_data["position"],
            "corners": patch_data["corners"],
        }

        # Save patch image
        cv2.imwrite(os.path.join(output_dir, f"patch_{i:04d}.png"), patch)

        # Save metadata
        with open(os.path.join(output_dir, f"patch_{i:04d}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
