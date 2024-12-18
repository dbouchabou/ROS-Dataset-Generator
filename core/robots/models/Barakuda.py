# robots/models/barakuda.py
from core.robots.base_robot import MobileRobot, RobotConfig
from pydantic import Field
from typing import Dict, Any


class BarakudaConfig(RobotConfig):
    motor_rpm_topic: str = Field(
        ..., description="Motor RPM topic is required for Barakuda"
    )
    movement_threshold: float = Field(
        default=10.0, description="Threshold for movement detection"
    )


class Barakuda(MobileRobot):
    """
    Barakuda robot implementation.
    Handles movement detection based on motor RPM values.
    """

    def __init__(self, config: BarakudaConfig):
        super().__init__(config)
        self.config: BarakudaConfig = config

    def _validate_config(self):
        """Validate Barakuda-specific configuration requirements"""
        if not self.config.motor_rpm_topic:
            raise ValueError("Barakuda requires motor_rpm_topic in configuration")

    def determine_movement(self, rpm_data: Dict[str, Any]) -> str:
        """
        Determine robot movement direction based on motor RPMs.

        Args:
            rpm_data: Dictionary containing motor RPM values

        Returns:
            str: Movement direction ('forward', 'backward', or 'stationary')
        """
        motor_values = rpm_data["motor_values"]
        if len(motor_values) != 4:
            raise ValueError("Expected 4 motor values")

        left_motors = (motor_values[0] + motor_values[3]) / 2
        right_motors = (motor_values[1] + motor_values[2]) / 2
        avg_movement = (left_motors + right_motors) / 2

        if abs(avg_movement) < self.config.movement_threshold:
            return "stationary"
        return "forward" if avg_movement > 0 else "backward"
