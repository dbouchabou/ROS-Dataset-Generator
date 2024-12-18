from typing import Dict, List
from core.data_models import ImageData
from core.tools.depth import (
    visualize_depth_map,
    depth_to_grayscale,
    grayscale_to_3channel,
)

from core.pipeline.base import Node, NodeResult, NodeStatus


class DepthVisualizationNode(Node):
    """Node for visualizing depth maps."""

    def __init__(self, colormap: str = "inferno"):
        super().__init__(
            name="depth_visualization", inputs={"depth_maps"}, outputs={"frames"}
        )
        self.colormap = colormap

    async def execute(self, inputs: Dict[str, List[ImageData]]) -> NodeResult:
        try:
            if "depth_maps" not in inputs:
                raise ValueError("No depth maps provided")

            depth_maps = inputs["depth_maps"]
            if not isinstance(depth_maps, list):
                raise ValueError("Depth maps must be a list")

            frames = []
            for depth_data in depth_maps:
                if not isinstance(depth_data, ImageData):
                    raise ValueError(f"Invalid depth data type: {type(depth_data)}")

                if self.colormap == "grayscale":
                    vis = grayscale_to_3channel(depth_to_grayscale(depth_data.image))
                else:
                    vis = visualize_depth_map(depth_data.image, colormap=self.colormap)
                frames.append(ImageData(image=vis, timestamp=depth_data.timestamp))

            return NodeResult(status=NodeStatus.COMPLETED, data={"frames": frames})
        except Exception as e:
            return NodeResult(status=NodeStatus.FAILED, error=e)
