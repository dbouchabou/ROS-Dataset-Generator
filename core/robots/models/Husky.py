# robots/models/Husky.py
from core.robots.base_robot import MobileRobot, RobotConfig
from pydantic import Field
from typing import Dict, Any


class HuskyConfig(RobotConfig):
    cmd_vel_topic: str = Field(
        ..., description="Motor cmd vel topic is required for Husky"
    )
    speed_treshold: float = Field(
        default=0.02, description="Threshold for movement detection"
    )


class Husky(MobileRobot):
    """
    Husky robot implementation.
    Handles movement detection based on motor RPM values.
    """

    def __init__(self, config: HuskyConfig):
        super().__init__(config)
        self.config: HuskyConfig = config

    def _validate_config(self):
        """Validate Husky-specific configuration requirements"""
        if not self.config.cmd_vel_topic:
            raise ValueError("Husky requires cmd_vel_topic in configuration")

    def determine_movement(self, cmd_vel_data: Dict[str, Any]) -> str:
        """
        Determine if the robot is moving forward, backward, or not moving.

        Args:
            cmd_vel_data (dict): A single cmd_vel data point.

        Returns:
            str: 'forward', 'backward', or 'stationary'
        """
        linear_x = cmd_vel_data["linear"]["x"]
        if (
            linear_x > self.config.speed_treshold
        ):  # You may need to adjust this threshold
            return "forward"
        elif (
            linear_x < -self.config.speed_treshold
        ):  # You may need to adjust this threshold
            return "backward"
        else:
            return "stationary"
