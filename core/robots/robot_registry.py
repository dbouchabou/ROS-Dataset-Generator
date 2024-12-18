"""
Robot Registry Module for Dynamic Robot Class Management

This module implements a singleton registry pattern for managing mobile robot implementations
and their configurations. It provides functionality to:
- Maintain a central registry of robot implementations
- Auto-discover and register robot classes from Python files
- Associate robot implementations with their corresponding configurations
- Provide access to registered robots through a singleton interface

The registry is initialized by scanning a specified directory for robot implementations
that inherit from the MobileRobot base class.
"""

import os
import sys
import inspect
from pathlib import Path
import importlib.util
from typing import Dict, Type, Union, Optional, NoReturn
from rich.console import Console
from .base_robot import MobileRobot, RobotConfig


class RobotRegistry:
    """
    Singleton registry for managing mobile robot implementations.

    This class maintains two registries:
    1. robot_registry: Maps robot names to their implementation classes
    2. config_registry: Maps robot names to their configuration classes

    The singleton pattern ensures a single point of access to robot registrations
    throughout the application lifecycle.

    Attributes:
        _instance (Optional[RobotRegistry]): Singleton instance of the registry
        _initialized (bool): Flag to prevent multiple initializations
        robot_registry (Dict[str, Type[MobileRobot]]): Mapping of robot names to classes
        config_registry (Dict[str, Type[RobotConfig]]): Mapping of robot names to configs
    """

    _instance: Optional["RobotRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "RobotRegistry":
        """
        Create or return the singleton instance of RobotRegistry.

        Returns:
            RobotRegistry: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(RobotRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the registry dictionaries if not already initialized.
        This method is safe to call multiple times due to the _initialized flag.
        """
        if not self._initialized:
            self.robot_registry: Dict[str, Type[MobileRobot]] = {}
            self.config_registry: Dict[str, Type[RobotConfig]] = {}
            self._initialized = True

    @classmethod
    def register_robot(
        cls, name: str, robot_class: Type[MobileRobot], config_class: Type[RobotConfig]
    ) -> None:
        """
        Register a robot implementation and its configuration class.

        Args:
            name (str): Unique identifier for the robot
            robot_class (Type[MobileRobot]): Robot implementation class
            config_class (Type[RobotConfig]): Robot configuration class

        Note:
            If a robot with the same name exists, it will be overwritten
        """
        instance = cls()
        instance.robot_registry[name] = robot_class
        instance.config_registry[name] = config_class


def initialize_registry(robots_dir: Union[str, Path]) -> None:
    """
    Initialize the robot registry by scanning and registering available robots.

    This function performs the following steps:
    1. Validates the robots directory exists
    2. Registers the built-in Barakuda robot
    3. Scans for additional robot implementations in Python files
    4. Automatically registers discovered robot classes and their configs

    Args:
        robots_dir (Union[str, Path]): Directory path containing robot implementations

    Raises:
        FileNotFoundError: If the specified robots directory doesn't exist

    Example:
        >>> initialize_registry("path/to/robots")
        Scanning for robot classes in path/to/robots...
        ✓ Registered Barakuda robot
        ✓ Registered CustomRobot robot
    """
    console = Console()
    robots_dir = Path(robots_dir)

    # Validate directory exists
    if not robots_dir.exists():
        raise FileNotFoundError(f"Robots directory not found: {robots_dir}")

    console.print(f"[cyan]Scanning for robot classes in {robots_dir}...[/cyan]")

    # Register built-in Barakuda robot
    # from .models.Barakuda import Barakuda, BarakudaConfig

    # RobotRegistry.register_robot("Barakuda", Barakuda, BarakudaConfig)
    # console.print("[green]✓ Registered Barakuda robot[/green]")

    def _find_config_class(
        module: object, robot_class: Type[MobileRobot]
    ) -> Optional[Type[RobotConfig]]:
        """
        Helper function to find the corresponding config class in a module.

        Args:
            module (object): The imported module to search
            robot_class (Type[MobileRobot]): The robot class to find config for

        Returns:
            Optional[Type[RobotConfig]]: Matching config class or None if not found
        """
        return next(
            (
                c
                for n, c in inspect.getmembers(module)
                if inspect.isclass(c)
                and issubclass(c, RobotConfig)
                and c != RobotConfig
            ),
            None,
        )

    # Scan for additional robot implementations
    for file_path in robots_dir.glob("*.py"):
        # Skip private files and Barakuda (already registered)
        if file_path.name.startswith("_"):
            # or file_path.stem == "Barakuda":
            continue

        try:
            # Import the module dynamically
            module_name = f"core.robots.models.{file_path.stem}"
            module = importlib.import_module(module_name)

            # Find and register robot classes
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, MobileRobot)
                    and obj != MobileRobot
                    and not inspect.isabstract(obj)
                ):
                    # Find corresponding config class
                    config_class = _find_config_class(module, obj)
                    if config_class:
                        RobotRegistry.register_robot(name, obj, config_class)
                        console.print(f"[green]✓ Registered {name} robot[/green]")

        except Exception as e:
            console.print(
                f"[yellow]⚠ Failed to load {file_path.name}: {str(e)}[/yellow]"
            )
