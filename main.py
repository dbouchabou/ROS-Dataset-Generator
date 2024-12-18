# main.py

import argparse
import asyncio
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Optional

from core.video.video_config import VideoGenerationConfig
from core.video.video_generation_manager import VideoGenerationManager
from core.data_models import CommandLineArgs, DatasetConfig
from core.tools.bag import (
    read_rgb_image_data,
    read_depth_image_data,
    read_odometry_data,
    read_cmd_vel_data,
    read_motor_rpm,
    estimate_frame_rate,
)
from core.tools.robots import (
    combine_odometry_cmdvel,
    combine_odometry_rpm,
    create_robot_instance,
)
from core.robots.base_robot import MobileRobot
from core.robots.robot_registry import initialize_registry

console = Console()


def parse_arguments() -> CommandLineArgs:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process ROS bag files for video generation and/or patch extraction."
    )

    # Required positional arguments
    parser.add_argument(
        "config_file", help="Path to the dataset configuration JSON file"
    )

    parser.add_argument("output_dir", help="Output directory for videos and configs")

    # Optional flags
    parser.add_argument(
        "--generate-video", action="store_true", help="Generate video of projected path"
    )

    parser.add_argument(
        "--generate-patches", action="store_true", help="Generate and save patches"
    )

    # Video config argument
    parser.add_argument(
        "--video-config",
        help="Path to the video generation configuration YAML file (required if --generate-video is specified)",
    )

    args = parser.parse_args()

    # Validation
    if args.generate_video and not args.video_config:
        parser.error("--video-config is required when --generate-video is specified")

    if not args.generate_video and args.video_config:
        parser.error(
            "--video-config should only be provided when --generate-video is specified"
        )

    if not (args.generate_video or args.generate_patches):
        parser.error(
            "At least one of --generate-video or --generate-patches must be specified"
        )

    try:
        return CommandLineArgs(
            config_file=args.config_file,
            output_dir=args.output_dir,
            video_config=args.video_config,
            generate_video=args.generate_video,
            generate_patches=args.generate_patches,
        )
    except ValueError as e:
        parser.error(str(e))


async def process_patches(
    dataset_config: DatasetConfig, robot: MobileRobot, bag_file: dict, output_dir: str
) -> None:
    """
    Process and generate patches from a bag file.

    Args:
        dataset_config: Dataset configuration
        robot: Robot instance
        bag_file: Dictionary containing bag file information
        output_dir: Output directory for patches
    """
    # Implementation of patch generation
    # This is a placeholder - implement according to your needs
    pass


async def process_bag_file(
    robot: MobileRobot,
    bag_file: dict,
    video_manager: Optional[VideoGenerationManager],
) -> None:
    """
    Process a single bag file for video generation.

    The function handles different data reading requirements:
    - RGB data: Required for RGB video and/or Depth Anything processing
    - Raw depth data: Only required if raw depth video or normal maps are enabled
    - Odometry data: Required for trajectory visualization

    Video type dependencies:
    - RGB video: Requires RGB image data
    - Depth Anything: Requires RGB image data (processes RGB to depth)
    - Raw depth video: Requires depth sensor data
    - Normal maps from raw depth: Requires depth sensor data
    - Normal maps from Depth Anything: Requires RGB image data

    Args:
        robot: Robot instance containing topic configurations
        bag_file: Dictionary containing bag file path and label
        video_manager: Optional video generation manager for video processing
    """
    bag_file_path = bag_file["path"]
    bag_file_name = os.path.splitext(os.path.basename(bag_file_path))[0]

    console.print(f"[bold blue]\nProcessing bag file:[/bold blue] {bag_file_path}")

    try:
        # Initialize data variables
        image_data_list = None
        depth_data_list = None

        if video_manager:
            # Determine which data types are needed
            need_rgb = any(
                (
                    settings.enabled
                    and (
                        settings.type == "rgb"  # RGB video
                        or video_name == "depth_anything"  # Depth Anything processing
                        or video_name
                        == "depth_anything_normal"  # Normal maps from Depth Anything
                    )
                )
                for video_name, settings in video_manager.config.videos.items()
            )

            need_raw_depth = any(
                (
                    settings.enabled
                    and (
                        video_name == "depth"  # Raw depth video
                        or video_name == "normal"  # Normal maps from raw depth
                    )
                )
                for video_name, settings in video_manager.config.videos.items()
            )

            need_odom = (
                video_manager.config.draw_trajectories
                or video_manager.config.draw_patches
            )

            # Read RGB data if needed
            if need_rgb:
                image_data_list = read_rgb_image_data(
                    bag_file_path, robot.config.image_topic
                )
                console.print("[green]Successfully read RGB data[/green]")

            # Read depth data only if raw depth processing is needed
            if need_raw_depth:
                depth_data_list = read_depth_image_data(
                    bag_file_path, robot.config.depth_topic
                )
                console.print("[green]Successfully read depth data[/green]")

        # Process odometry data
        combined_data = []
        if need_odom:
            if robot.config.odom_topic is not None:
                odometry_data = read_odometry_data(
                    bag_file_path, robot.config.odom_topic
                )

                # Combine with cmd_vel or motor_rpm based on availability
                if robot.config.cmd_vel_topic is not None:
                    cmd_vel_data = read_cmd_vel_data(
                        bag_file_path, robot.config.cmd_vel_topic
                    )
                    combined_data = combine_odometry_cmdvel(odometry_data, cmd_vel_data)
                elif robot.config.motor_rpm_topic is not None:
                    motor_rpm_data = read_motor_rpm(
                        bag_file_path, robot.config.motor_rpm_topic
                    )
                    combined_data = combine_odometry_rpm(odometry_data, motor_rpm_data)

            console.print("[green]Successfully processed odometry data[/green]")

        # Generate videos if video manager is available
        if video_manager:
            # Estimate frame rate if not provided in configuration
            if video_manager.config.frame_rate is not None:
                estimated_frame_rate = video_manager.config.frame_rate
            else:
                estimated_frame_rate = estimate_frame_rate(
                    bag_file_path, robot.config.image_topic
                )

            await video_manager.process_and_generate_videos(
                bag_file_name=bag_file_name,
                image_data=image_data_list,
                depth_data=depth_data_list,
                combined_data=combined_data,
                robot_config=robot.config,
                estimated_frame_rate=estimated_frame_rate,
            )

    except Exception as e:
        console.print(
            f"[bold red]Error processing bag file {bag_file_path}:[/bold red] {str(e)}"
        )
        console.print_exception()
        raise


async def main():
    console.print(Panel("[bold blue]ROS Bag Processing Script[/bold blue]"))

    try:
        # Parse arguments
        cmd_args = parse_arguments()

        # Load dataset configuration
        dataset_config = DatasetConfig.from_json(cmd_args.config_file)

        # Print processing information
        table = Table(title="Processing Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Dataset Config", cmd_args.config_file)
        table.add_row("Output Directory", cmd_args.output_dir)
        table.add_row("Generate Video", str(cmd_args.generate_video))
        table.add_row("Generate Patches", str(cmd_args.generate_patches))
        if cmd_args.generate_video:
            table.add_row("Video Config", cmd_args.video_config)
        console.print(table)

        # Initialize video manager if generating videos
        video_manager = None
        if cmd_args.generate_video:
            video_config = VideoGenerationConfig.model_validate_yaml(
                cmd_args.video_config
            )
            video_manager = VideoGenerationManager(
                config=video_config, output_dir=cmd_args.output_dir
            )
            video_manager.setup_output_directories(dataset_config.dataset_name)
            video_manager.save_configurations(
                dataset_config=dataset_config,
                robot_config_path=f"robots_configs/{dataset_config.robot_config}.json",
                video_config_path=cmd_args.video_config,
            )

        # Initialize registry and robot
        initialize_registry(Path("core/robots/models"))
        robot = create_robot_instance(
            config_path=f"robots_configs/{dataset_config.robot_config}.json",
            robot_type=dataset_config.robot_config,
        )

        # Process each bag file
        total_files = len(dataset_config.bag_files)
        for idx, bag_file in enumerate(dataset_config.bag_files, 1):
            console.print(
                f"\n[bold cyan]Processing file {idx}/{total_files}[/bold cyan]"
            )

            if cmd_args.generate_patches:
                await process_patches(
                    dataset_config, robot, bag_file, cmd_args.output_dir
                )

            if cmd_args.generate_video:
                await process_bag_file(
                    robot,
                    bag_file,
                    video_manager,
                )

        console.print(
            "\n[bold green]✨✨ All Bags Processing complete! ✨✨[/bold green]"
        )

    except Exception as e:
        console.print(f"[bold red]Error in main:[/bold red] {str(e)}")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
