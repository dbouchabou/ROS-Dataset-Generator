# video/video_generation_manager.py

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
from datetime import datetime
import yaml
import json

from rich.console import Console

from core.data_models import DatasetConfig, ImageData
from .video_config import VideoGenerationConfig
from .factory import VideoPipelineFactory

# Setup logging
logger = logging.getLogger(__name__)
console = Console()


class VideoGenerationManager:
    """
    Manages video generation using pipeline-based processing.

    This class orchestrates the video generation process by:
    1. Setting up the processing environment
    2. Creating and configuring pipelines
    3. Managing pipeline execution and dependencies
    4. Handling configuration and output management
    """

    def __init__(self, config: VideoGenerationConfig, output_dir: str):
        """
        Initialize the video generation manager.

        Args:
            config: Video generation configuration
            output_dir: Base directory for outputs
        """
        self.config = config
        self.output_base = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_pipeline_factory = None
        self.dataset_output_dir = None
        self.video_dir = None
        self.config_dir = None

    def setup_output_directories(self, dataset_name: str) -> None:
        """
        Set up output directory structure.

        Creates the following structure:
        output_dir/
        └── dataset_{dataset_name}/
            ├── videos/        # Generated video files
            └── configs/       # Configuration files

        Args:
            dataset_name: Name of the dataset being processed
        """
        self.dataset_output_dir = self.output_base / f"dataset_{dataset_name}"
        self.video_dir = self.dataset_output_dir / "videos"
        self.config_dir = self.dataset_output_dir / "configs"

        # Create directories
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pipeline factory
        self.video_pipeline_factory = VideoPipelineFactory(
            config=self.config, output_dir=self.video_dir
        )

    def save_configurations(
        self,
        dataset_config: DatasetConfig,
        robot_config_path: str,
        video_config_path: str,
    ) -> None:
        """
        Save all configuration files for reproducibility.

        Args:
            dataset_config: Dataset configuration
            robot_config_path: Path to robot configuration file
            video_config_path: Path to video generation configuration file
        """
        if not self.config_dir:
            raise RuntimeError("Output directories not initialized")

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
        self._save_processing_summary(dataset_config)

        console.print(f"[green]Configuration files saved to:[/green] {self.config_dir}")

    def _save_processing_summary(self, dataset_config: DatasetConfig) -> None:
        """
        Save detailed processing summary.

        Args:
            dataset_config: Dataset configuration to include in summary
        """
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
        Process data and generate videos using pipelines.

        Args:
            bag_file_name: Name of the bag file being processed
            image_data: RGB image data
            depth_data: Depth image data
            combined_data: Combined odometry data
            robot_config: Robot configuration
            estimated_frame_rate: Optional frame rate from bag file
        """
        if not self.video_pipeline_factory:
            raise RuntimeError("Pipeline factory not initialized")

        # Update frame rate in configuration
        if estimated_frame_rate:
            self.config.frame_rate = estimated_frame_rate
            console.print(f"[cyan]Using frame rate: {estimated_frame_rate} fps[/cyan]")
        else:
            console.print(
                f"[yellow]Using default frame rate: {self.config.frame_rate} fps[/yellow]"
            )

        # Create pipelines
        self.video_pipeline_factory.create_pipelines()

        # Prepare initial data
        initial_data = {
            "rgb_images": image_data if image_data else [],
            "depth_images": depth_data if depth_data else [],
            "odometry_data": combined_data,
            "robot_config": robot_config,
            "file_name": bag_file_name,
        }

        # Process shared Depth Anything pipeline if needed
        # depth_maps = None
        shared_pipeline = False

        if self.video_pipeline_factory._needs_depth_anything_processing():
            shared_pipeline = True
            try:
                console.print(
                    "[cyan]Processing Depth Anything shared pipeline...[/cyan]"
                )
                results = await self.video_pipeline_factory.registry.execute_pipeline(
                    "depth_anything_shared_video", initial_data
                )

                # Extract depth maps from results if successful
                if results and "depth_maps" in results:
                    result = results["depth_maps"]
                    if result.is_success and result.data:
                        depth_maps = result.data.get("depth_maps")
                        # Add depth maps to initial data for dependent pipelines
                        initial_data["depth_maps"] = depth_maps

            except Exception as e:
                logger.error(f"Error in Depth Anything shared pipeline: {e}")
                console.print(f"Node depth_anything failed: {e}")

        # Process each enabled video type
        for video_name, settings in self.config.videos.items():
            if not settings.enabled:
                continue

            if shared_pipeline and video_name in [
                "depth_anything",
                "depth_anything_normal",
            ]:
                # Skip dependent pipelines if shared pipeline was processed
                continue

            console.print(f"Generating {video_name} video...")

            try:
                pipeline_name = {
                    "rgb": "rgb_video",
                    "depth": "raw_depth_video",
                    "depth_anything": "depth_anything_video",
                    "depth_anything_normal": "depth_anything_normal_video",
                    "normal": "normal_video",
                }.get(video_name)

                if not pipeline_name:
                    logger.warning(f"No pipeline defined for video type: {video_name}")
                    continue

                # Execute pipeline
                results = await self.video_pipeline_factory.registry.execute_pipeline(
                    pipeline_name, initial_data
                )

                # Process pipeline results
                if results:
                    for key, result in results.items():
                        if result.is_success:
                            if result.data:
                                # Update initial data with successful results
                                initial_data.update(result.data)
                        else:
                            logger.error(f"Node {key} failed: {result.error}")
                            console.print(f"Node {key} failed: {result.error}")

            except Exception as e:
                logger.exception(f"Error generating {video_name} video: {e}")
                console.print(f"Error generating {video_name} video: {e}")
                continue

        console.print("\n[bold green]🎞️ 🎞️ Videos Processing complete! 🎞️ 🎞️[/bold green]")
