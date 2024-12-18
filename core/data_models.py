# data_models.py

import os
import json
import numpy as np
from typing import List, Optional, Dict
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field, model_validator


class CommandLineArgs(BaseModel):
    """
    Command line arguments validation model.

    This model validates the command-line arguments passed to the script,
    ensuring all required arguments are present and valid.

    Attributes:
        config_file: Path to the dataset configuration JSON file
        output_dir: Directory where videos and configs will be saved
        generate_video: Flag to enable video generation
        generate_patches: Flag to enable patch generation
        video_config: Path to video generation YAML configuration

    Note:
        At least one of generate_video or generate_patches must be True.
        video_config is required only when generate_video is True.
    """

    config_file: str = Field(
        ..., description="Path to the dataset configuration JSON file"
    )
    output_dir: str = Field(..., description="Output directory for videos and configs")
    generate_video: bool = Field(
        default=False, description="Generate video of projected path"
    )
    generate_patches: bool = Field(
        default=False, description="Generate and save patches"
    )
    video_config: Optional[str] = Field(
        None, description="Path to the video generation config YAML file"
    )

    @model_validator(mode="after")
    def validate_args(self) -> "CommandLineArgs":
        """
        Validate command line arguments combination.

        Validates:
        1. At least one generation option is enabled
        2. video_config is present when needed
        3. video_config is not provided when not needed

        Returns:
            CommandLineArgs: Self if validation passes

        Raises:
            ValueError: If validation fails
        """
        if not (self.generate_video or self.generate_patches):
            raise ValueError(
                "At least one of --generate-video or --generate-patches must be specified."
            )

        if self.generate_video and not self.video_config:
            raise ValueError(
                "video_config is required when --generate-video is specified."
            )

        if not self.generate_video and self.video_config:
            raise ValueError(
                "video_config should only be provided when --generate-video is specified."
            )

        return self


class DatasetConfig(BaseSettings):
    """
    Dataset configuration model.

    Handles loading and validating dataset configuration from JSON files.
    Manages information about the dataset, robot configuration, and bag files.

    Attributes:
        config_file: Path to the configuration file
        robot_config: Robot configuration name/type
        bag_files: List of dictionaries containing bag file information
        dataset_name: Derived from config_file name if not explicitly set
    """

    config_file: str
    robot_config: str
    bag_files: List[Dict[str, str]]
    dataset_name: Optional[str] = None

    @model_validator(mode="after")
    def set_dataset_name(self) -> "DatasetConfig":
        """
        Set dataset_name from config_file if not explicitly provided.

        Extracts the dataset name from the config file path by taking
        the filename without extension.

        Returns:
            DatasetConfig: Self with dataset_name set
        """
        if self.config_file and not self.dataset_name:
            self.dataset_name = os.path.splitext(os.path.basename(self.config_file))[0]
        return self

    def __str__(self) -> str:
        """
        Create a human-readable string representation.

        Returns:
            str: Formatted string containing dataset information
        """
        return (
            f"Dataset name: {self.dataset_name}\n"
            f"Robot config: {self.robot_config}\n"
            f"Number of bag files: {len(self.bag_files)}\n"
            f"Bag files:\n"
            + "\n".join(
                f"  - {bag['path']} (Label: {bag['label']})" for bag in self.bag_files
            )
        )

    @classmethod
    def from_json(cls, config_file: str) -> "DatasetConfig":
        """
        Create a DatasetConfig instance from a JSON file.

        Args:
            config_file: Path to the JSON configuration file

        Returns:
            DatasetConfig: Instantiated and validated configuration

        Raises:
            FileNotFoundError: If config_file doesn't exist
            JSONDecodeError: If JSON parsing fails
            ValidationError: If config data is invalid
        """
        with open(config_file, "r") as f:
            config_data = json.load(f)
        config_data["config_file"] = config_file  # Add config_file to the data
        return cls(**config_data)


class ImageData(BaseModel):
    """
    Image data container with timestamp.

    Holds an image array and its associated timestamp from the ROS bag.
    Supports numpy arrays through pydantic's arbitrary types.

    Attributes:
        image: Numpy array containing the image data
        timestamp: ROS timestamp of the image

    Note:
        Uses arbitrary_types_allowed to support numpy arrays
    """

    image: np.ndarray
    timestamp: float

    class Config:
        """Enable arbitrary types to support numpy arrays."""

        arbitrary_types_allowed = True

    def __str__(self) -> str:
        """
        Create a human-readable string representation.

        Returns:
            str: Description including image shape and timestamp
        """
        return f"Image(shape={self.image.shape}, timestamp={self.timestamp})"


class Vector3D(BaseModel):
    """
    Represents a 3D vector with x, y, z components.

    Attributes:
        x: X component of the vector
        y: Y component of the vector
        z: Z component of the vector
    """

    x: float = Field(description="X component of the vector")
    y: float = Field(description="Y component of the vector")
    z: float = Field(description="Z component of the vector")


class Quaternion(BaseModel):
    """
    Represents a quaternion for 3D orientation.

    Attributes:
        x: X component of the quaternion
        y: Y component of the quaternion
        z: Z component of the quaternion
        w: W component of the quaternion (scalar part)
    """

    x: float = Field(description="X component of the quaternion")
    y: float = Field(description="Y component of the quaternion")
    z: float = Field(description="Z component of the quaternion")
    w: float = Field(description="W component of the quaternion (scalar part)")

    @model_validator(mode="after")
    def validate_normalization(self) -> "Quaternion":
        """
        Validate that the quaternion is approximately normalized.

        Returns:
            The validated Quaternion instance

        Raises:
            ValueError: If the quaternion magnitude deviates significantly from 1.0
        """
        norm = np.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
        if not np.isclose(norm, 1.0, rtol=1e-3):
            raise ValueError(f"Quaternion is not normalized (magnitude: {norm})")
        return self


# class OdometryData(BaseModel):
#     """
#     Container for synchronized odometry data combining pose and twist information.

#     This class represents a complete odometry state including position, orientation,
#     linear and angular velocities, and their associated covariances. It is designed
#     to be compatible with ROS nav_msgs/Odometry message format.

#     Attributes:
#         timestamp: Unix timestamp of the odometry data
#         position: 3D position in meters (x, y, z)
#         orientation: Quaternion orientation (x, y, z, w)
#         linear_velocity: Linear velocity in m/s (x, y, z)
#         angular_velocity: Angular velocity in rad/s (x, y, z)
#         pose_covariance: 6x6 covariance matrix for pose (position and orientation)
#         twist_covariance: Optional 6x6 covariance matrix for twist (linear and angular velocity)

#     Example:
#         >>> odom_data = OdometryData(
#         ...     timestamp=1234567890.123,
#         ...     position={'x': 1.0, 'y': 2.0, 'z': 0.0},
#         ...     orientation={'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
#         ...     linear_velocity={'x': 0.5, 'y': 0.0, 'z': 0.0},
#         ...     angular_velocity={'x': 0.0, 'y': 0.0, 'z': 0.1},
#         ...     pose_covariance=[0.0] * 36
#         ... )
#     """

#     model_config = ConfigDict(
#         frozen=True,  # Make instances immutable
#         validate_assignment=True,  # Validate values on assignment
#         json_schema_extra={
#             "examples": [
#                 {
#                     "timestamp": 1234567890.123,
#                     "position": {"x": 1.0, "y": 2.0, "z": 0.0},
#                     "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
#                     "linear_velocity": {"x": 0.5, "y": 0.0, "z": 0.0},
#                     "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.1},
#                     "pose_covariance": [0.0] * 36,
#                 }
#             ]
#         },
#     )

#     timestamp: float = Field(description="Unix timestamp of the odometry data")
#     position: Vector3D = Field(description="3D position in meters")
#     orientation: Quaternion = Field(description="Quaternion orientation")
#     linear_velocity: Vector3D = Field(description="Linear velocity in m/s")
#     angular_velocity: Vector3D = Field(description="Angular velocity in rad/s")
#     pose_covariance: List[float] = Field(
#         description="6x6 covariance matrix for pose (position and orientation)",
#         min_items=36,
#         max_items=36,
#     )
#     twist_covariance: Optional[List[float]] = Field(
#         None,
#         description="Optional 6x6 covariance matrix for twist (linear and angular velocity)",
#         min_items=36,
#         max_items=36,
#     )

#     @model_validator(mode="after")
#     def validate_covariance_matrices(self) -> "OdometryData":
#         """
#         Validate the structure of covariance matrices.

#         Ensures that:
#         1. Pose covariance is a valid 6x6 symmetric matrix
#         2. Twist covariance, if provided, is a valid 6x6 symmetric matrix

#         Returns:
#             The validated OdometryData instance

#         Raises:
#             ValueError: If covariance matrices are invalid
#         """
#         # Validate pose covariance
#         pose_cov = np.array(self.pose_covariance).reshape(6, 6)
#         if not np.allclose(pose_cov, pose_cov.T):
#             raise ValueError("Pose covariance matrix is not symmetric")

#         # Validate twist covariance if provided
#         if self.twist_covariance is not None:
#             twist_cov = np.array(self.twist_covariance).reshape(6, 6)
#             if not np.allclose(twist_cov, twist_cov.T):
#                 raise ValueError("Twist covariance matrix is not symmetric")

#         return self

#     def get_position_array(self) -> np.ndarray:
#         """
#         Get position as a numpy array.

#         Returns:
#             3D numpy array of position [x, y, z]
#         """
#         return np.array([self.position.x, self.position.y, self.position.z])

#     def get_orientation_array(self) -> np.ndarray:
#         """
#         Get orientation as a numpy array.

#         Returns:
#             4D numpy array of quaternion [x, y, z, w]
#         """
#         return np.array(
#             [
#                 self.orientation.x,
#                 self.orientation.y,
#                 self.orientation.z,
#                 self.orientation.w,
#             ]
#         )

#     def get_twist_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Get linear and angular velocities as numpy arrays.

#         Returns:
#             Tuple of (linear_velocity [x, y, z], angular_velocity [x, y, z])
#         """
#         linear = np.array(
#             [self.linear_velocity.x, self.linear_velocity.y, self.linear_velocity.z]
#         )
#         angular = np.array(
#             [self.angular_velocity.x, self.angular_velocity.y, self.angular_velocity.z]
#         )
#         return linear, angular

#     class Config:
#         """Pydantic model configuration."""

#         arbitrary_types_allowed = True  # Allow numpy arrays
#         json_encoders = {
#             np.ndarray: lambda x: x.tolist()  # Handle numpy arrays in JSON
#         }
