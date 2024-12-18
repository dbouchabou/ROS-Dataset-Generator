# video_generation_manager.py

"""
Video Generation Manager for ROS bag processing.

This module provides a manager class that handles:
- Video generation from ROS bag data
- Configuration management and validation
- Output directory organization
- Multiple video types (RGB, depth, normal maps)
- Depth Anything processing
"""

import shutil
from datetime import datetime
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from rich.console import Console
import json

from data_models import DatasetConfig, ImageData
from tools.depth import apply_depth_anything_v2
from tools.image import convert_depth_images_to_normal_images
from video.video_generation import generate_video
from video.video_config import VideoGenerationConfig, VideoType, VideoSettings

console = Console()


class VideoGenerationManager:
    """
    Manages video generation from ROS bag data.

    This class handles all aspects of video generation including:
    - Directory setup and configuration management
    - Processing of different video types (RGB, depth, normal maps)
    - Depth Anything model integration
    - Video file generation with configurable parameters

    Attributes:
        config (VideoGenerationConfig): Configuration for video generation
        output_base (Path): Base output directory path
        timestamp (str): Current timestamp for file naming
        dataset_output_dir (Path): Dataset-specific output directory
        video_dir (Path): Directory for generated videos
        config_dir (Path): Directory for configuration files
    """

    def __init__(self, config: VideoGenerationConfig, output_dir: str):
        """
        Initialize the VideoGenerationManager.

        Args:
            config: Configuration object containing video generation settings
            output_dir: Base directory for all outputs
        """
        self.config = config
        self.output_base = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def setup_output_directories(self, dataset_name: str) -> None:
        """
        Set up output directories for videos and configuration files.

        Creates the following structure:
        output_dir/
        └── dataset_{dataset_name}/
            ├── videos/         # Generated video files
            └── configs/        # Configuration and summary files

        Args:
            dataset_name: Name of the dataset being processed
        """
        self.dataset_output_dir = self.output_base / f"dataset_{dataset_name}"
        self.video_dir = self.dataset_output_dir / "videos"
        self.config_dir = self.dataset_output_dir / "configs"

        # Create directories
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def save_configurations(
        self,
        dataset_config: DatasetConfig,
        robot_config_path: str,
        video_config_path: str,
    ) -> None:
        """
        Save all configuration files for reproducibility.

        Saves:
        1. Dataset configuration (JSON)
        2. Robot configuration (copy with timestamp)
        3. Video generation configuration (YAML)
        4. Processing summary (text file)

        Args:
            dataset_config: Dataset configuration object
            robot_config_path: Path to robot configuration file
            video_config_path: Path to video generation configuration file
        """
        # Save dataset configuration
        dataset_config_path = self.config_dir / "dataset_config.json"
        with open(dataset_config_path, "w") as f:
            json.dump(dataset_config.model_dump(), f, indent=2)

        # Copy and timestamp robot configuration
        robot_config_dest = self.config_dir / f"robot_config_{self.timestamp}.json"
        shutil.copy2(robot_config_path, robot_config_dest)

        # Save video generation configuration
        video_config_dest = self.config_dir / f"video_config_{self.timestamp}.yaml"
        with open(video_config_dest, "w") as f:
            yaml.dump(self.config.model_dump(), f, indent=2)

        # Create detailed processing summary
        summary_path = self.config_dir / "processing_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Processing Summary - {self.timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {dataset_config.dataset_name}\n")
            f.write(f"Robot Config: {dataset_config.robot_config}\n")
            f.write(f"Number of Bag Files: {len(dataset_config.bag_files)}\n\n")
            f.write("Enabled Video Types:\n")
            for name, settings in self.config.videos.items():
                if settings.enabled:
                    f.write(f"- {name} ({settings.type})\n")
            f.write("\nBag Files:\n")
            for bag in dataset_config.bag_files:
                f.write(f"- {bag['path']} (Label: {bag['label']})\n")

        console.print(f"[green]Configuration files saved to:[/green] {self.config_dir}")

    async def process_depth_anything(
        self,
        image_data_list: List[ImageData],
        depth_anything_settings: VideoSettings,
        depth_anything_normal_settings: VideoSettings,
    ) -> Tuple[List[ImageData], List[ImageData]]:
        """
        Process RGB images through Depth Anything model.

        Workflow:
        1. Extract model configuration
        2. Process RGB images to generate depth maps
        3. Optionally convert depth maps to normal maps
        4. Handle errors and provide feedback

        Args:
            image_data_list: List of RGB images with timestamps
            depth_anything_settings: Depth Anything model configuration
            depth_anything_normal_settings: Settings for normal map generation

        Returns:
            Tuple containing:
            - List of depth images (empty if processing fails)
            - List of normal maps (empty if not requested or processing fails)
        """
        try:
            # Extract model configuration
            model_config = {}
            if depth_anything_settings.depth_model_config:
                model_config = {
                    "encoder": depth_anything_settings.depth_model_config.encoder,
                    "batch_size": depth_anything_settings.depth_model_config.batch_size,
                    "num_processes": depth_anything_settings.depth_model_config.num_processes,
                }

            # Process depth
            console.print("[cyan]Processing images with Depth Anything...[/cyan]")
            depth_data = await apply_depth_anything_v2(image_data_list, **model_config)

            normal_data = []

            if depth_data:
                console.print("[green]Successfully generated depth images[/green]")
                # Generate normal maps if requested
                if depth_anything_normal_settings.enabled:
                    console.print("[cyan]Converting depth to normal images...[/cyan]")
                    normal_data = convert_depth_images_to_normal_images(depth_data)
                    console.print("[green]Successfully generated normal images[/green]")
                return depth_data, normal_data
            else:
                console.print("[yellow]No depth data generated[/yellow]")
                return [], []

        except Exception as e:
            console.print(
                f"[bold red]Error processing Depth Anything:[/bold red] {str(e)}"
            )
            console.print_exception()
            return [], []

    def generate_videos(
        self,
        bag_file_name: str,
        data_collections: Dict[str, List[ImageData]],
        combined_data: List,
        robot_config: Any,
        estimated_frame_rate: Optional[int] = None,
    ) -> None:
        """
        Generate videos for all enabled video types.

        Frame rate priority:
        1. Use estimated frame rate from bag file if available
        2. Fall back to default 30 fps

        Args:
            bag_file_name: Name of the bag file (for output naming)
            data_collections: Dictionary mapping video types to image data
            combined_data: Combined odometry data for visualization
            robot_config: Robot configuration
            estimated_frame_rate: Optional frame rate from bag file
        """
        frame_rate = estimated_frame_rate or 30

        if estimated_frame_rate:
            console.print(f"[cyan]Using frame rate: {frame_rate} fps[/cyan]")
        else:
            console.print(
                f"[yellow]Using default frame rate: {frame_rate} fps[/yellow]"
            )

        # Process each enabled video type
        for video_name, settings in self.config.videos.items():
            if not settings.enabled:
                continue

            video_filename = (
                f"{bag_file_name}_path_projection_{settings.filename_suffix}.mp4"
            )

            data = data_collections.get(video_name, [])
            if not data:
                console.print(
                    f"[yellow]Warning: No data available for {video_name} video[/yellow]"
                )
                continue

            console.print(f"[bold green]Generating {video_name} video...[/bold green]")

            try:
                generate_video(
                    image_data_list=data,
                    odometry_data=combined_data,
                    config=robot_config,
                    output_path=str(self.video_dir),
                    video_filename=video_filename,
                    frame_rate=frame_rate,
                    D=self.config.path_parameters.D,
                    d=self.config.path_parameters.d,
                    depth_image=(settings.type == VideoType.DEPTH),
                    depth_colormap=(
                        settings.depth_colormap
                        if settings.type == VideoType.DEPTH
                        else "grayscale"
                    ),
                    draw_patches=self.config.draw_patches,
                    draw_trajectories=self.config.draw_trajectories,
                )
            except Exception as e:
                console.print(
                    f"[bold red]Error generating {video_name} video:[/bold red] {str(e)}"
                )
                console.print_exception()
                continue

    async def process_and_generate_videos(
        self,
        bag_file_name: str,
        image_data: List[ImageData],
        depth_data: List[ImageData],
        combined_data: List,
        robot_config: Any,
        estimated_frame_rate: Optional[int] = None,
    ) -> None:
        """
        Main method to process and generate all video types.

        Workflow:
        1. Process RGB data if available
        2. Process depth data and generate normal maps if needed
        3. Run Depth Anything processing if enabled
        4. Generate all enabled video types

        Args:
            bag_file_name: Name of the bag file being processed
            image_data: RGB image data from bag
            depth_data: Depth image data from bag
            combined_data: Combined odometry data
            robot_config: Robot configuration
            estimated_frame_rate: Optional frame rate from bag file
        """
        data_collections = {}

        # Process RGB data
        if image_data:
            data_collections["rgb"] = image_data

        # Process depth and normal data
        if depth_data:
            data_collections["depth"] = depth_data
            if self.config.videos.get("normal", {}).enabled:
                console.print(
                    "[cyan]Converting depth images to normal images...[/cyan]"
                )
                data_collections["normal"] = convert_depth_images_to_normal_images(
                    depth_data
                )

        # Check if Depth Anything processing is needed
        need_depth_anything = (
            self.config.videos.get("depth_anything", {}).enabled
            or self.config.videos.get("depth_anything_normal", {}).enabled
        )

        if need_depth_anything and image_data:
            depth_anything_settings = self.config.videos.get("depth_anything")
            depth_anything_normal_settings = self.config.videos.get(
                "depth_anything_normal"
            )

            console.print("[bold cyan]Processing Depth Anything...[/bold cyan]")
            depth_anything_data, depth_anything_normal = (
                await self.process_depth_anything(
                    image_data, depth_anything_settings, depth_anything_normal_settings
                )
            )

            if depth_anything_data:
                # Store depth data only if explicitly enabled
                if depth_anything_settings and depth_anything_settings.enabled:
                    data_collections["depth_anything"] = depth_anything_data

                # Store normal data if enabled
                if (
                    depth_anything_normal_settings
                    and depth_anything_normal_settings.enabled
                ):
                    console.print("[cyan]Storing Depth Anything normal maps...[/cyan]")
                    data_collections["depth_anything_normal"] = depth_anything_normal

        # Generate all enabled videos
        self.generate_videos(
            bag_file_name=bag_file_name,
            data_collections=data_collections,
            combined_data=combined_data,
            robot_config=robot_config,
            estimated_frame_rate=estimated_frame_rate,
        )
