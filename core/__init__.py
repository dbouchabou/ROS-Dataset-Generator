"""
ROS Bag Processing and Video Generation Library

This library provides tools for processing ROS bag files and generating videos,
with support for different robot configurations, depth processing, and visualization.

The library consists of several key components:
- Video generation and processing
- Dataset configuration and management
- Robot models and registration
- Depth processing and visualization
- Utility functions for geometry, transforms, and data processing

Typical usage example:

    from rosbag_processor import DatasetConfig, create_robot_instance, generate_video
    
    # Load configuration
    config = DatasetConfig.from_json("dataset_config.json")
    
    # Create robot instance
    robot = create_robot_instance("robot_config.json", "Barakuda")
    
    # Generate video
    generate_video(image_data, odometry_data, config, output_path="output")
"""

from importlib.metadata import version

# Version information
try:
    __version__ = version("rosbag_processor")
except ImportError:  # pragma: no cover
    __version__ = "unknown"

# Core data models
from .data_models import (
    CommandLineArgs,
    DatasetConfig,
    ImageData,
)

# Video generation and processing
from .video.video_generation import generate_video
from .video.video_config import VideoGenerationConfig, VideoType, VideoSettings
from .video.video_generation_manager import VideoGenerationManager

# Robot-related imports
from .robots.base_robot import (
    MobileRobot,
    RobotConfig,
    SensorValue,
)
from .robots.robot_registry import (
    RobotRegistry,
    initialize_registry,
)

# Robot model implementations
from .robots.models.Barakuda import Barakuda, BarakudaConfig
from .robots.models.Husky import Husky, HuskyConfig
from .robots.models.Aru import Aru, AruConfig

# Deep learning models
from .models.depth_anything_v2.dpt import DepthAnythingV2

# Utility functions
from .tools import (
    # Geometry and transforms
    transform,
    # Image processing
    image,
    # Depth processing
    depth,
    # Data processing
    data_processing,
    # ROS bag handling
    bag,
    # Robot utilities
    robots,
)

# Patch extraction and processing
from .patch import patch_extraction

# Public API
__all__ = [
    # Version
    "__version__",
    # Core data models
    "CommandLineArgs",
    "DatasetConfig",
    "ImageData",
    # Video generation
    "generate_video",
    "VideoGenerationConfig",
    "VideoType",
    "VideoSettings",
    "VideoGenerationManager",
    # Robot base classes
    "MobileRobot",
    "RobotConfig",
    "SensorValue",
    # Robot registry
    "RobotRegistry",
    "initialize_registry",
    "create_robot_instance",
    # Robot implementations
    "Barakuda",
    "BarakudaConfig",
    "Husky",
    "HuskyConfig",
    "Aru",
    "AruConfig",
    # Deep learning
    "DepthAnythingV2",
    # Utility modules
    "transform",
    "image",
    "depth",
    "data_processing",
    "bag",
    "robots",
    # Patch processing
    "patch_extraction",
]


# Initialize robot registry on import
def _initialize():
    """Initialize the robot registry with available robot implementations."""
    try:
        from pathlib import Path

        initialize_registry(Path(__file__).parent / "robots" / "models")
    except Exception as e:  # pragma: no cover
        import warnings

        warnings.warn(f"Failed to initialize robot registry: {e}")


_initialize()
