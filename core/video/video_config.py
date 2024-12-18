# video/video_config.py

"""
Configuration models for video generation from ROS bag files.

This module defines the configuration structure for generating different types
of videos from ROS bag data, including:
- RGB videos from camera feeds
- Depth maps (raw or from Depth Anything model)
- Normal maps (from depth data)

The configuration is typically loaded from a YAML file and validated using
Pydantic models.
"""

from enum import Enum
from typing import Dict, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator
import yaml
from pathlib import Path


class VideoType(str, Enum):
    """
    Supported video types.

    Attributes:
        RGB: Standard RGB video from camera
        DEPTH: Depth map visualization
        NORMAL: Surface normal map visualization
    """

    RGB = "rgb"
    DEPTH = "depth"
    NORMAL = "normal"


class DepthAnythingConfig(BaseModel):
    """
    Configuration for Depth Anything model processing.

    Controls the behavior of the Depth Anything model for
    generating depth maps from RGB images.

    Attributes:
        encoder: Model architecture ('vitb' for base or 'vitl' for large)
        batch_size: Number of images to process at once
        num_processes: Number of parallel processing threads
    """

    model_config = ConfigDict(frozen=False)

    encoder: str = Field(
        default="vitl",
        description="Encoder type to use ('vitb' for base, 'vitl' for large)",
    )
    batch_size: int = Field(
        default=5, description="Number of images to process in each batch", gt=0
    )
    num_processes: int = Field(
        default=16, description="Number of parallel processes for data processing", gt=0
    )

    @model_validator(mode="after")
    def validate_encoder(self) -> "DepthAnythingConfig":
        """
        Validate encoder type selection.

        Raises:
            ValueError: If encoder is not 'vitb' or 'vitl'
        """
        valid_encoders = ["vitb", "vitl"]
        if self.encoder not in valid_encoders:
            raise ValueError(f"Encoder must be one of {valid_encoders}")
        return self


class PathParameters(BaseModel):
    """
    Parameters for robot path projection visualization.

    Controls how the robot's projected path is displayed in videos.

    Attributes:
        D: Length of projected path in front of robot (meters)
        d: Spacing between points along path (meters)
    """

    model_config = ConfigDict(frozen=False)

    D: float = Field(
        default=1.0,
        description="Robot path length in front of the robot (meters)",
        gt=0,
    )
    d: float = Field(
        default=0.1,
        description="Distance between points of the robot path (meters)",
        gt=0,
    )

    @model_validator(mode="after")
    def validate_path_parameters(self) -> "PathParameters":
        """
        Validate path parameter relationships.

        Ensures point spacing is smaller than total path length.

        Raises:
            ValueError: If d >= D
        """
        if self.d >= self.D:
            raise ValueError("Point distance (d) must be smaller than path length (D)")
        return self


class VideoSettings(BaseModel):
    """
    Configuration for a specific video type.

    Controls the generation settings for each type of video output.

    Attributes:
        enabled: Whether to generate this video type
        type: Type of video (RGB, DEPTH, or NORMAL)
        filename_suffix: Suffix for output filename
        depth_colormap: Colormap for depth visualization
        depth_model_config: Configuration for Depth Anything processing
    """

    model_config = ConfigDict(frozen=False)

    enabled: bool = Field(
        default=True, description="Whether to generate this video type"
    )
    type: VideoType = Field(
        ..., description="Type of video to generate (rgb, depth, or normal)"
    )
    filename_suffix: str = Field(..., description="Suffix to append to output filename")
    depth_colormap: Optional[str] = Field(
        default=None,
        description="Colormap for depth visualization (inferno, viridis, plasma, magma)",
    )
    depth_model_config: Optional[DepthAnythingConfig] = Field(
        default=None, description="Configuration for Depth Anything model"
    )

    @model_validator(mode="after")
    def validate_depth_settings(self) -> "VideoSettings":
        """
        Validate depth-specific settings.

        Ensures:
        1. Depth videos have a colormap specified
        2. Non-depth videos don't have colormaps
        3. Colormap is a valid matplotlib option

        Raises:
            ValueError: If validation fails
        """
        if self.type == VideoType.DEPTH and not self.depth_colormap:
            raise ValueError("depth_colormap must be set for depth videos")

        if self.type != VideoType.DEPTH and self.depth_colormap:
            raise ValueError("depth_colormap should only be set for depth videos")

        if self.depth_colormap and self.depth_colormap not in [
            "inferno",
            "viridis",
            "plasma",
            "magma",
        ]:
            raise ValueError(
                "depth_colormap must be one of: inferno, viridis, plasma, magma"
            )
        return self


class VideoGenerationConfig(BaseModel):
    """
    Main configuration for video generation.

    Controls all aspects of video generation including:
    - Frame rate and visualization options
    - Path visualization parameters
    - Settings for each video type

    Attributes:
        frame_rate: Optional override for video frame rate
        draw_trajectories: Whether to draw robot trajectories
        draw_patches: Whether to draw patch areas
        path_parameters: Robot path visualization settings
        videos: Dictionary of settings for each video type
    """

    model_config = ConfigDict(frozen=False)

    frame_rate: Optional[int] = Field(
        default=None,
        description="Frame rate override. If None, uses auto-detection from bag file",
        gt=0,
    )
    draw_trajectories: bool = Field(
        default=True, description="Whether to draw robot trajectories on videos"
    )
    draw_patches: bool = Field(
        default=True, description="Whether to draw patch areas on videos"
    )
    path_parameters: PathParameters = Field(
        ..., description="Parameters for robot path projection"
    )
    videos: Dict[str, VideoSettings] = Field(
        ..., description="Configuration for each video type"
    )

    @model_validator(mode="after")
    def validate_video_configs(self) -> "VideoGenerationConfig":
        """
        Validate the complete video configuration.

        Ensures:
        1. At least one video type is enabled
        2. Depth Anything dependencies are properly configured
        3. Required configurations are present

        Returns:
            Self with validated and potentially modified configuration

        Raises:
            ValueError: If validation fails
        """
        # Ensure at least one video type is enabled
        if not any(video.enabled for video in self.videos.values()):
            raise ValueError("At least one video type must be enabled")

        # Validate depth anything configurations
        depth_anything = self.videos.get("depth_anything")
        depth_anything_normal = self.videos.get("depth_anything_normal")

        # Handle depth_anything_normal dependencies
        if depth_anything_normal and depth_anything_normal.enabled:
            if not depth_anything:
                # Create depth_anything config if needed
                self.videos["depth_anything"] = VideoSettings(
                    enabled=False,  # Keep disabled for normal-only generation
                    type=VideoType.DEPTH,
                    filename_suffix="DepthAnythingV2",
                    depth_colormap="inferno",
                    depth_model_config=DepthAnythingConfig(),
                )
            elif not depth_anything.depth_model_config:
                depth_anything.depth_model_config = DepthAnythingConfig()

        return self

    def get_video_settings(self, video_type: str) -> Optional[VideoSettings]:
        """
        Safely get settings for a specific video type.

        Args:
            video_type: Name of the video type

        Returns:
            VideoSettings if found, None otherwise
        """
        return self.videos.get(video_type)

    @classmethod
    def model_validate_yaml(
        cls, yaml_path: Union[str, Path]
    ) -> "VideoGenerationConfig":
        """
        Load and validate configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Validated VideoGenerationConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML parsing or validation fails
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            try:
                config_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")

        try:
            return cls.model_validate(config_data)
        except ValueError as e:
            raise ValueError(f"Invalid configuration: {e}")

    def save(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            config_path: Path where to save the configuration

        Raises:
            ValueError: If saving fails
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, "w") as f:
                yaml.dump(self.model_dump(), f, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")
