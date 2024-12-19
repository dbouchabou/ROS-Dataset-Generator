# video/factory.py

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .video_config import VideoGenerationConfig, VideoType, VideoSettings
from core.pipeline.base import Pipeline, PipelineRegistry, Node

from core.pipeline.nodes import (
    RGBPreprocessNode,
    DepthPreprocessNode,
    DepthAnythingNode,
    DepthVisualizationNode,
    NormalMapNode,
    TrajectoryVisualizationNode,
    PatchVisualizationNode,
    VideoEncoderNode,
    AddInfoNode,
)


class VideoPipelineFactory:
    """Factory for creating video processing pipelines."""

    def __init__(self, config: VideoGenerationConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.registry = PipelineRegistry()

    def create_pipelines(self):
        """
        Create all required pipelines based on configuration.

        This method analyzes the video configuration and creates appropriate
        pipelines for each enabled video type, handling dependencies between
        different processing stages.
        """
        # Check for Depth Anything dependencies
        # Create shared Depth Anything pipeline if needed
       
        if self._needs_depth_anything_processing():
            print("SHARED")
            self._create_depth_anything_shared_pipeline()
        else:
            print("NOT SHARED")

        # Create pipelines for each enabled video type
        for video_name, settings in self.config.videos.items():
            if not settings.enabled:
                continue

            if video_name == "rgb":
                print("create rgb pipeline")
                self._create_rgb_pipeline(settings)
            if video_name == "depth":
                print("create depth pipeline")
                self._create_raw_depth_pipeline(settings)
            if video_name == "normal":
                print("create normal pipeline")
                self._create_normal_pipeline(settings)

            if (
                video_name
                == "depth_anything"
                # and not self._needs_depth_anything_processing()
            ):
                print("only depthanything")
                self._create_depth_anything_pipeline(settings)

            if (
                video_name
                == "depth_anything_normal"
                # and not self._needs_depth_anything_processing()
            ):
                print("only normal")
                self._create_depth_anything_normal_pipeline(settings)

    def _needs_depth_anything_processing(self) -> bool:
        """Check if Depth Anything processing is needed."""
        # return all(
        #     settings.enabled
        #     and (video_name in ["depth_anything", "depth_anything_normal"])
        #     for video_name, settings in self.config.videos.items()
        # )

        if (
            self.config.videos["depth_anything"].enabled
            and self.config.videos["depth_anything_normal"].enabled
        ):
            return True
        return False

    def _create_depth_anything_shared_pipeline(self):
        """Create shared pipeline for Depth Anything processing."""

        settings_depth_anything = self.config.videos["depth_anything"]
        settings_depth_anything_normal = self.config.videos["depth_anything_normal"]

        nodes = []

        nodes.append(RGBPreprocessNode())

        nodes.append(
            DepthAnythingNode(model_config=settings_depth_anything.depth_model_config)
        )

        nodes.append(
            DepthVisualizationNode(colormap=settings_depth_anything.depth_colormap)
        )

        if self.config.draw_trajectories:
            nodes.append(TrajectoryVisualizationNode(self.config))
        else:
            print("Trajectory visualization is disabled")

        if self.config.draw_patches:
            nodes.append(PatchVisualizationNode(self.config))
        else:
            print("Patch visualization is disabled")

        nodes.append(AddInfoNode())

        nodes.append(
            VideoEncoderNode(
                output_path=str(self.output_dir),
                filename=f"_{settings_depth_anything.filename_suffix}.mp4",
                fps=self.config.frame_rate,
            )
        )

        nodes.append(NormalMapNode())

        if self.config.draw_trajectories:
            nodes.append(TrajectoryVisualizationNode(self.config))
        else:
            print("Trajectory visualization is disabled")

        if self.config.draw_patches:
            nodes.append(PatchVisualizationNode(self.config))
        else:
            print("Patch visualization is disabled")

        nodes.append(AddInfoNode())

        nodes.append(
            VideoEncoderNode(
                output_path=str(self.output_dir),
                filename=f"_{settings_depth_anything_normal.filename_suffix}.mp4",
                fps=self.config.frame_rate,
            )
        )

        pipeline = Pipeline(
            name="depth_anything_shared_video",
            nodes=nodes,
            expected_inputs={
                "rgb_images",
                "robot_config",
                "odometry_data",
                "file_name",
            },
        )
        self.registry.register_pipeline(pipeline)

    def _create_rgb_pipeline(self, settings: VideoSettings):
        nodes = []

        # Add visualization nodes and get final frame key
        # vis_nodes, final_frame_key = self._create_visualization_nodes(settings)
        # nodes.extend(vis_nodes)

        # Create encoder with correct input key
        # nodes.append(self._create_video_encoder_node(settings))

        nodes.append(RGBPreprocessNode())

        if self.config.draw_trajectories:
            nodes.append(TrajectoryVisualizationNode(self.config))
        else:
            print("Trajectory visualization is disabled")

        if self.config.draw_patches:
            nodes.append(PatchVisualizationNode(self.config))
        else:
            print("Patch visualization is disabled")

        nodes.append(AddInfoNode())

        nodes.append(
            VideoEncoderNode(
                output_path=str(self.output_dir),
                filename=f"_{settings.filename_suffix}.mp4",
                fps=self.config.frame_rate,
            )
        )

        pipeline = Pipeline(
            name="rgb_video",
            nodes=nodes,
            expected_inputs={
                "rgb_images",
                "odometry_data",
                "robot_config",
                "file_name",
            },
        )
        self.registry.register_pipeline(pipeline)

    def _create_depth_anything_pipeline(self, settings: VideoSettings):
        """Create pipeline for Depth Anything video generation."""
        nodes = []

        # Add visualization nodes if enabled
        # nodes.extend(self._create_visualization_nodes(settings))

        # Add encoder as final node
        # nodes.append(self._create_video_encoder_node(settings))

        nodes.append(RGBPreprocessNode())

        nodes.append(DepthAnythingNode(model_config=settings.depth_model_config))

        nodes.append(DepthVisualizationNode(colormap=settings.depth_colormap))

        if self.config.draw_trajectories:
            nodes.append(TrajectoryVisualizationNode(self.config))
        else:
            print("Trajectory visualization is disabled")

        if self.config.draw_patches:
            nodes.append(PatchVisualizationNode(self.config))
        else:
            print("Patch visualization is disabled")

        nodes.append(AddInfoNode())

        nodes.append(
            VideoEncoderNode(
                output_path=str(self.output_dir),
                filename=f"_{settings.filename_suffix}.mp4",
                fps=self.config.frame_rate,
            )
        )

        pipeline = Pipeline(
            name="depth_anything_video",
            nodes=nodes,
            expected_inputs={
                "rgb_images",
                "odometry_data",
                "robot_config",
                "file_name",
            },
        )
        self.registry.register_pipeline(pipeline)

    def _create_depth_anything_normal_pipeline(self, settings: VideoSettings):
        """Create pipeline for normal maps from Depth Anything."""
        nodes = []

        nodes.append(RGBPreprocessNode())

        nodes.append(DepthAnythingNode(model_config=settings.depth_model_config))

        nodes.append(NormalMapNode())

        if self.config.draw_trajectories:
            nodes.append(TrajectoryVisualizationNode(self.config))
        else:
            print("Trajectory visualization is disabled")

        if self.config.draw_patches:
            nodes.append(PatchVisualizationNode(self.config))
        else:
            print("Patch visualization is disabled")

        nodes.append(AddInfoNode())

        nodes.append(
            VideoEncoderNode(
                output_path=str(self.output_dir),
                filename=f"_{settings.filename_suffix}.mp4",
                fps=self.config.frame_rate,
            )
        )

        pipeline = Pipeline(
            name="depth_anything_normal_video",
            nodes=nodes,
            expected_inputs={
                "rgb_images",
                "odometry_data",
                "robot_config",
                "file_name",
            },
        )
        self.registry.register_pipeline(pipeline)

    def _create_raw_depth_pipeline(self, settings: VideoSettings):
        """Create pipeline for raw depth video processing."""
        nodes = []

        nodes.append(DepthPreprocessNode())

        nodes.append(DepthVisualizationNode(colormap=settings.depth_colormap))

        if self.config.draw_trajectories:
            nodes.append(TrajectoryVisualizationNode(self.config))
        else:
            print("Trajectory visualization is disabled")

        if self.config.draw_patches:
            nodes.append(PatchVisualizationNode(self.config))
        else:
            print("Patch visualization is disabled")

        nodes.append(AddInfoNode())

        nodes.append(
            VideoEncoderNode(
                output_path=str(self.output_dir),
                filename=f"_{settings.filename_suffix}.mp4",
                fps=self.config.frame_rate,
            )
        )

        pipeline = Pipeline(
            name="raw_depth_video",
            nodes=nodes,
            expected_inputs={
                "depth_images",
                "odometry_data",
                "robot_config",
                "file_name",
            },
        )
        self.registry.register_pipeline(pipeline)

    def _create_normal_pipeline(self, settings: VideoSettings):
        """Create pipeline for normal maps from raw depth."""
        nodes = []

        nodes.append(DepthPreprocessNode())

        nodes.append(NormalMapNode())

        if self.config.draw_trajectories:
            nodes.append(TrajectoryVisualizationNode(self.config))
        else:
            print("Trajectory visualization is disabled")

        if self.config.draw_patches:
            nodes.append(PatchVisualizationNode(self.config))
        else:
            print("Patch visualization is disabled")

        nodes.append(AddInfoNode())

        nodes.append(
            VideoEncoderNode(
                output_path=str(self.output_dir),
                filename=f"_{settings.filename_suffix}.mp4",
                fps=self.config.frame_rate,
            )
        )

        pipeline = Pipeline(
            name="normal_video",
            nodes=nodes,
            expected_inputs={
                "depth_images",
                "odometry_data",
                "robot_config",
                "file_name",
            },
        )
        self.registry.register_pipeline(pipeline)
