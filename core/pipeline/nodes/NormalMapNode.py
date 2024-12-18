from typing import Dict, List
from core.data_models import ImageData

from core.tools.image import (
    convert_depth_images_to_normal_images,
)

from core.pipeline.base import Node, NodeResult, NodeStatus


class NormalMapNode(Node):
    """Node for generating normal maps from depth maps."""

    def __init__(self):
        super().__init__(
            name="normal_map",
            inputs={"depth_maps"},
            outputs={"frames"},  # Changed to "frames" to match trajectory input
        )

    async def execute(self, inputs: Dict[str, List[ImageData]]) -> NodeResult:
        try:
            frames = convert_depth_images_to_normal_images(inputs["depth_maps"])

            return NodeResult(status=NodeStatus.COMPLETED, data={"frames": frames})
        except Exception as e:
            return NodeResult(status=NodeStatus.FAILED, error=e)
