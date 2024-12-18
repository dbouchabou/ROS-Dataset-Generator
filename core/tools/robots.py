import os
import json
import numpy as np
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.console import Console
from rich.panel import Panel
from typing import Union, Type, List, Dict, Any
from pathlib import Path

from core.tools.transform import (
    apply_rigid_motion,
    camera_frame_to_image,
)

from core.tools.data_processing import (
    find_closest_cmd_vel,
)

from core.robots.base_robot import MobileRobot
from core.robots.base_robot import RobotConfig
from core.robots.robot_registry import RobotRegistry


def odom_to_robot(odometry_data, WORLD_TO_ROBOT):
    """
    Converts odometry data from world coordinates to robot frame coordinates using a given transformation matrix.

    Args:
    odometry_data (list): List of odometry data points, where each point is a dictionary containing 'position' and 'timestamp'.
    WORLD_TO_ROBOT (np.array): A 4x4 transformation matrix representing the rigid motion from the world frame to the robot frame.

    Returns:
    list: A list of odometry data points with positions transformed to the robot frame.
    """
    odom_into_robot = []

    for odom in odometry_data:
        position = odom["position"]
        linear = odom["linear"]

        # Create a numpy array for the position in world coordinates
        point_world_position = np.array([position["x"], position["y"], position["z"]])

        # Compute the point's coordinates in the robot frame
        point_robot_position = apply_rigid_motion(point_world_position, WORLD_TO_ROBOT)

        # Create a numpy array for the position in world coordinates
        point_world_linear = np.array([linear["x"], linear["y"], linear["z"]])

        # Compute the point's coordinates in the robot frame
        point_robot_linear = apply_rigid_motion(
            point_world_linear, WORLD_TO_ROBOT, rot_only=True
        )

        # Create a dictionary for the transformed position and the timestamp
        point_robot_dict = {
            "position": {
                "x": point_robot_position[0][0],
                "y": point_robot_position[0][1],
                "z": point_robot_position[0][2],
            },
            "linear": {
                "x": point_robot_linear[0][0],
                "y": point_robot_linear[0][1],
                "z": point_robot_linear[0][2],
            },
            "timestamp": odom["timestamp"],
            "movement_direction": odom["movement_direction"],
        }

        # Append the transformed point to the list
        odom_into_robot.append(point_robot_dict)

    return odom_into_robot


def robot_to_image(odometry_data, ROBOT_TO_CAM, config):
    odom_into_image = []

    for odom in odometry_data:

        position = odom["position"]

        points_robot = np.array(
            [
                position["x"],
                position["y"],
                position["z"],
            ]
        )

        # Compute the points coordinates in the camera frame
        points_camera = apply_rigid_motion(points_robot, ROBOT_TO_CAM)

        # Compute the points coordinates in the image plan
        points_image = camera_frame_to_image(points_camera, config.K)

        points_image = {
            "position": {"x": points_image[0][0], "y": points_image[0][1]},
            "timestamp": odom["timestamp"],
            "movement_direction": odom["movement_direction"],
        }

        odom_into_image.append(points_image)
    return odom_into_image


def odom_to_image(odometry_data, WORLD_TO_ROBOT, ROBOT_TO_CAM, K):
    """
    Convert odometry data from world coordinates to image coordinates.

    This function takes a list of odometry data points in world coordinates and transforms
    them through a series of coordinate frames (world -> robot -> camera -> image) to obtain
    their corresponding positions in image coordinates.

    Args:
        odometry_data (list): List of dictionaries containing odometry data.
            Each dictionary should have 'position' (with 'x', 'y', 'z' keys) and 'timestamp'.
        WORLD_TO_ROBOT (np.ndarray): 4x4 transformation matrix from world to robot frame.
        ROBOT_TO_CAM (np.ndarray): 4x4 transformation matrix from robot to camera frame.
        K (np.ndarray): 3x3 camera intrinsic matrix.

    Returns:
        list: List of dictionaries containing transformed odometry data in image coordinates.
            Each dictionary contains 'position' (with 'x', 'y' keys) and 'timestamp'.

    Note:
        This function assumes the existence of helper functions 'apply_rigid_motion' and
        'camera_frame_to_image' which are not defined in this scope.
    """
    odom_into_image = []

    for odom in odometry_data:
        position = odom["position"]
        point_world = np.array([position["x"], position["y"], position["z"]])

        points_robot = apply_rigid_motion(point_world, WORLD_TO_ROBOT)
        points_camera = apply_rigid_motion(points_robot, ROBOT_TO_CAM)
        points_image = camera_frame_to_image(points_camera, K)

        odom_into_image.append(
            {
                "position": {"x": points_image[0][0], "y": points_image[0][1]},
                "timestamp": odom["timestamp"],
            }
        )

    return odom_into_image


def calculate_wheel_trajectories(center_trajectory, wheel_distance=0.65):
    """
    Calculate the trajectories of the left and right wheels based on the center trajectory of a robot.

    This function takes the center trajectory of a robot and computes the corresponding
    trajectories for the left and right wheels, given the distance between the wheels.
    It handles various edge cases and potential errors in the input data.

    Parameters:
    center_trajectory (list): A list of dictionaries representing the center trajectory points.
                              Each dictionary should have the following keys:
                              - 'position': A dictionary with 'x', 'y', and 'z' coordinates
                              - 'timestamp': The timestamp of the point
                              - 'movement_direction': The direction of movement at this point

    wheel_distance (float, optional): The distance between the left and right wheels in meters.
                                      Defaults to 0.65 meters.

    Returns:
    tuple: Two lists of dictionaries (left_trajectory, right_trajectory), where each dictionary
           represents a point on the respective wheel's trajectory and has the same structure
           as the input center_trajectory points.

    Notes:
    - The function calculates wheel positions by creating perpendicular vectors to the
      direction of movement at each point.
    - Points with zero direction vectors or other invalid calculations are skipped.
    - NaN values in calculations result in skipping the affected points.
    - Errors during processing of individual points are caught and logged, allowing the
      function to continue processing subsequent points.
    - The z-coordinate is preserved from the center trajectory for each wheel point.

    Warnings and errors are printed to the console for:
    - Zero direction vectors
    - Invalid norms during vector normalization
    - Invalid perpendicular vectors
    - NaN values in calculated wheel positions

    Example:
    >>> center_traj = [
    ...     {"position": {"x": 0, "y": 0, "z": 0}, "timestamp": 0, "movement_direction": "forward"},
    ...     {"position": {"x": 1, "y": 1, "z": 0}, "timestamp": 1, "movement_direction": "forward"}
    ... ]
    >>> left_traj, right_traj = calculate_wheel_trajectories(center_traj, wheel_distance=0.5)
    """
    left_trajectory = []
    right_trajectory = []

    for i in range(len(center_trajectory)):
        try:
            current_point = np.array(
                [
                    center_trajectory[i]["position"]["x"],
                    center_trajectory[i]["position"]["y"],
                    center_trajectory[i]["position"]["z"],
                ]
            )

            # Calculate direction vector
            if i < len(center_trajectory) - 1:
                next_point = np.array(
                    [
                        center_trajectory[i + 1]["position"]["x"],
                        center_trajectory[i + 1]["position"]["y"],
                        center_trajectory[i + 1]["position"]["z"],
                    ]
                )
                direction = next_point - current_point
            elif i > 0:
                # For the last point, use the previous direction
                direction = current_point - np.array(
                    [
                        center_trajectory[i - 1]["position"]["x"],
                        center_trajectory[i - 1]["position"]["y"],
                        center_trajectory[i - 1]["position"]["z"],
                    ]
                )
            else:
                # If there's only one point, skip it
                continue

            # Check for zero direction vector
            if np.all(direction == 0):
                print(
                    f"Warning: Zero direction vector at index {i}. Skipping this point."
                )
                continue

            # Normalize direction vector
            norm = np.linalg.norm(direction)
            if norm == 0 or np.isnan(norm):
                print(f"Warning: Invalid norm at index {i}. Skipping this point.")
                continue
            direction = direction / norm

            # Calculate perpendicular vector in the xy-plane
            perpendicular = np.array([-direction[1], direction[0], 0])
            perp_norm = np.linalg.norm(perpendicular)
            if perp_norm == 0 or np.isnan(perp_norm):
                print(
                    f"Warning: Invalid perpendicular vector at index {i}. Skipping this point."
                )
                continue
            perpendicular = perpendicular / perp_norm

            # Calculate left and right wheel positions
            left_point = current_point + (wheel_distance / 2) * perpendicular
            right_point = current_point - (wheel_distance / 2) * perpendicular

            # Check for NaN values
            if np.isnan(left_point).any() or np.isnan(right_point).any():
                print(
                    f"Warning: NaN values encountered at index {i}. Skipping this point."
                )
                continue

            # Add to trajectories
            left_trajectory.append(
                {
                    "position": {
                        "x": float(left_point[0]),
                        "y": float(left_point[1]),
                        "z": float(left_point[2]),
                    },
                    "timestamp": center_trajectory[i]["timestamp"],
                    "movement_direction": center_trajectory[i]["movement_direction"],
                }
            )
            right_trajectory.append(
                {
                    "position": {
                        "x": float(right_point[0]),
                        "y": float(right_point[1]),
                        "z": float(right_point[2]),
                    },
                    "timestamp": center_trajectory[i]["timestamp"],
                    "movement_direction": center_trajectory[i]["movement_direction"],
                }
            )

        except Exception as e:
            print(f"Error processing point at index {i}: {e}")
            continue

    return left_trajectory, right_trajectory


def determine_movement_direction(cmd_vel_data, speed_treshold=0.02):
    """
    Determine if the robot is moving forward, backward, or not moving.

    Args:
        cmd_vel_data (dict): A single cmd_vel data point.

    Returns:
        str: 'forward', 'backward', or 'stationary'
    """
    linear_x = cmd_vel_data["linear"]["x"]
    if linear_x > speed_treshold:  # You may need to adjust this threshold
        return "forward"
    elif linear_x < -speed_treshold:  # You may need to adjust this threshold
        return "backward"
    else:
        return "stationary"


def determine_robot_movement_2(rpm_data, threshold=10.0):
    """
    Determine robot movement direction based on motor RPMs.

    Args:
        rpm_data (Float32MultiArray): Motor RPM data with 4 values
        threshold (float): Minimum RPM to consider as movement

    Returns:
        str: 'forward', 'backward', or 'stationary'
    """

    motor_values = rpm_data["motor_values"]
    # Ensure we have 4 motor values
    if len(motor_values) != 4:
        raise ValueError("Expected 4 motor values")

    # Calculate average RPM for left and right sides
    left_motors = (motor_values[0] + motor_values[3]) / 2
    right_motors = (motor_values[1] + motor_values[2]) / 2

    # Calculate overall movement direction
    avg_movement = (left_motors + right_motors) / 2

    if abs(avg_movement) < threshold:
        return "stationary"
    elif avg_movement > 0:
        return "forward"
    else:
        return "backward"


def combine_odometry_cmdvel(
    odometry_data: List[Dict[str, Any]],
    cmd_vel_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Combine odometry and cmd_vel.

    Args:
        odometry_data (List[Dict[str, Any]]): List of odometry data dictionaries.
        cmd_vel_data (List[Dict[str, Any]]): List of cmd_vel data dictionaries.

    Returns:
        List[Dict[str, Any]]: Combined data list.
    """
    console = Console()
    combined_data = []
    closest_cmd_vel = {}

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]📦 Combining odometry and cmd_vel data",
            total=len(odometry_data),
        )
        for odom in odometry_data:
            closest_cmd_vel = find_closest_cmd_vel(odom["timestamp"], cmd_vel_data)
            direction = determine_movement_direction(closest_cmd_vel)
            combined_data.append(
                {
                    **odom,
                    "movement_direction": direction,
                    "cmd_vel_timestamp": closest_cmd_vel["timestamp"],
                }
            )
            progress.update(task, advance=1)

    if len(combined_data) != len(odometry_data):
        console.print(
            f"[bold red]Error:[/bold red] Size mismatch: combined_data ({len(combined_data)}) != odometry_data ({len(odometry_data)})"
        )
        return None

    return combined_data


def combine_odometry_rpm(
    odometry_data: List[Dict[str, Any]],
    rpm_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Combine odometry and motor RPM data.

    Args:
        odometry_data: List of odometry data dictionaries.
        rpm_data: List of motor RPM data dictionaries.

    Returns:
        Combined data list.
    """
    console = Console()
    combined_data = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]📦 Combining odometry and motor RPM data",
            total=len(odometry_data),
        )

        for odom in odometry_data:

            closest_rpm = find_closest_cmd_vel(odom["timestamp"], rpm_data)
            movement = determine_robot_movement_2(closest_rpm, threshold=100.0)

            combined_data.append(
                {
                    **odom,
                    "movement_direction": movement,
                }
            )
            progress.update(task, advance=1)

    if len(combined_data) != len(odometry_data):
        console.print(
            f"[bold red]Error:[/bold red] Size mismatch: combined_data ({len(combined_data)}) != odometry_data ({len(odometry_data)})"
        )
        return None

    return combined_data


def load_robot_config(
    config_path: Union[str, Path],
    config_class: Type[RobotConfig] = RobotConfig,
    robot_type: str = "generic",
) -> RobotConfig:
    """
    Load and validate robot configuration from a JSON file.

    Args:
        config_path: Path to the configuration file
        config_class: Configuration class to use (default: RobotConfig)
        robot_type: Type of robot being configured (for display purposes)

    Returns:
        Validated robot configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
        pydantic.ValidationError: If config doesn't match schema
    """
    console = Console()

    # Convert to Path object for better path handling
    config_path = Path(config_path)

    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        # Load JSON config
        with open(config_path, "r") as f:
            config_data = json.load(f)

        # Create configuration object
        config = config_class(**config_data)

        # Display loaded configuration
        console.print(
            Panel(
                f"Loaded {robot_type} configuration from {config_path.name}\n"
                f"Topics configured:\n"
                f"  Image: {config.image_topic or 'Not configured'}\n"
                f"  Depth: {config.depth_topic or 'Not configured'}\n"
                f"  Odometry: {config.odom_topic or 'Not configured'}\n"
                f"  Motor RPM: {config.motor_rpm_topic or 'Not configured'}\n"
                f"Camera parameters: {'Configured' if config.has_camera_params() else 'Not configured'}",
                title=f"[bold green]{robot_type.title()} Configuration[/bold green]",
                border_style="green",
            )
        )

        return config

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

    except Exception as e:
        console.print(f"[bold red]Error loading configuration:[/bold red] {str(e)}")
        raise


def setup_robot_imports():
    """Set up the Python path to include the robots package."""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent  # Adjust this based on your project structure

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def get_robot_class(robot_name: str) -> Type[MobileRobot]:
    """Get robot class from registry."""
    registry = RobotRegistry()
    if robot_name not in registry.robot_registry:
        raise ValueError(f"Unknown robot type: {robot_name}")
    return registry.robot_registry[robot_name]


def get_config_class(robot_name: str) -> Type[RobotConfig]:
    """Get config class from registry."""
    registry = RobotRegistry()
    if robot_name not in registry.config_registry:
        raise ValueError(f"Unknown robot config type: {robot_name}")
    return registry.config_registry[robot_name]


def create_robot_instance(
    config_path: Union[str, Path],
    robot_type: str,
) -> MobileRobot:
    """Create a robot instance."""
    try:
        # Get appropriate classes from registry
        robot_class = get_robot_class(robot_type)
        config_class = get_config_class(robot_type)

        # Load configuration
        with open(config_path, "r") as f:
            config_data = json.load(f)

        config = config_class(**config_data)
        return robot_class(config)

    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error creating robot instance:[/bold red] {str(e)}")
        raise
