"""
Base classes and configurations for mobile robot implementations.

This module provides the foundational classes needed to implement mobile robot behaviors:
- RobotConfig: Configuration model for robot parameters and sensor topics
- MobileRobot: Abstract base class for mobile robot implementations

The module uses Pydantic for configuration validation and requires abstract method
implementation for robot movement determination.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypeVar, Union
from pydantic import BaseModel, Field
import numpy as np

# Type variable for sensor data values
SensorValue = TypeVar("SensorValue", np.ndarray, List[float], float, str)


class RobotConfig(BaseModel):
    """
    Configuration model for mobile robot parameters and ROS topics.

    This class handles robot configuration including camera parameters and ROS topics
    for sensors and control. It provides properties to access camera matrices and
    validation methods for camera parameters.

    Attributes:
        image_topic (Optional[str]): ROS topic for RGB camera images
        depth_topic (Optional[str]): ROS topic for depth camera data
        odom_topic (Optional[str]): ROS topic for odometry data
        cmd_vel_topic (Optional[str]): ROS topic for velocity commands
        camera_intrinsics (Optional[List[List[float]]]): 3x3 camera intrinsics matrix
        camera_extrinsics (Optional[List[List[float]]]): 4x4 camera extrinsics matrix
    """

    image_topic: Optional[str] = Field(
        default=None, description="ROS topic name for RGB camera images"
    )
    depth_topic: Optional[str] = Field(
        default=None, description="ROS topic name for depth camera data"
    )
    odom_topic: Optional[str] = Field(
        default=None, description="ROS topic name for odometry data"
    )
    cmd_vel_topic: Optional[str] = Field(
        default=None, description="ROS topic name for velocity commands"
    )
    motor_rpm_topic: Optional[str] = Field(
        default=None, description="ROS topic name for rpm motor values"
    )
    wheel_distance: Optional[float] = Field(
        default=1, description="Distance width between the robot's wheels"
    )
    camera_intrinsics: Optional[List[List[float]]] = Field(
        default=None,
        min_items=3,
        max_items=3,
        description="3x3 camera intrinsics matrix",
    )
    camera_extrinsics: Optional[List[List[float]]] = Field(
        default=None,
        min_items=4,
        max_items=4,
        description="4x4 camera extrinsics matrix representing camera to robot transform",
    )

    @property
    def K(self) -> Optional[np.ndarray]:
        """
        Get camera intrinsics matrix as NumPy array.

        Returns:
            Optional[np.ndarray]: 3x3 camera intrinsics matrix or None if not configured
        """
        return np.array(self.camera_intrinsics) if self.camera_intrinsics else None

    @property
    def cam_to_robot(self) -> Optional[np.ndarray]:
        """
        Get camera extrinsics (camera-to-robot transform) matrix as NumPy array.

        Returns:
            Optional[np.ndarray]: 4x4 transformation matrix or None if not configured
        """
        return np.array(self.camera_extrinsics) if self.camera_extrinsics else None

    def has_camera_params(self) -> bool:
        """
        Check if both camera intrinsics and extrinsics are configured.

        Returns:
            bool: True if both camera matrices are available, False otherwise
        """
        return self.camera_intrinsics is not None and self.camera_extrinsics is not None


class MobileRobot(ABC):
    """
    Abstract base class for mobile robot implementations.

    This class provides the basic structure for implementing mobile robot behaviors.
    Concrete implementations must provide the determine_movement method to define
    robot behavior based on sensor data.

    Attributes:
        config (RobotConfig): Configuration parameters for the robot
    """

    def __init__(self, config: RobotConfig):
        """
        Initialize the mobile robot with configuration parameters.

        Args:
            config (RobotConfig): Configuration parameters for the robot
        """
        self.config = config
        self._validate_config()
        self.rgb_data = None
        self.depth_data = None
        self.odom_data = None

    def _validate_config(self) -> None:
        """
        Validate robot-specific configuration requirements.

        This method should be overridden by concrete implementations to add
        specific configuration validation rules. The base implementation
        performs no validation.
        """
        pass

    @abstractmethod
    def determine_movement(self, sensor_data: Dict[str, SensorValue]) -> str:
        """
        Determine the robot's next movement based on sensor data.

        This abstract method must be implemented by concrete robot classes to define
        how the robot should move based on its sensor inputs.

        Args:
            sensor_data (Dict[str, SensorValue]): Dictionary of sensor names to their values.
                Values can be numpy arrays, lists, floats, or strings depending on sensor type.

        Returns:
            str: Command string indicating the determined movement

        Raises:
            NotImplementedError: If the concrete class doesn't implement this method
        """
        pass
