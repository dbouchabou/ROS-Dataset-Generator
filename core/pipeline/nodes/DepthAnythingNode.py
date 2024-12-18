from typing import Dict, List
from core.data_models import ImageData
from core.tools.depth import apply_depth_anything_v2

from core.pipeline.base import Node, NodeResult, NodeStatus


class DepthAnythingNode(Node):
    """Node for generating depth maps using Depth Anything."""

    def __init__(self, model_config: Dict):
        super().__init__(
            name="depth_anything", inputs={"rgb_images"}, outputs={"depth_maps"}
        )
        self.model_config = {
            "encoder": model_config.encoder,
            "batch_size": model_config.batch_size,
            "num_processes": model_config.num_processes,
        }

    async def execute(self, inputs: Dict[str, List[ImageData]]) -> NodeResult:
        try:
            depth_maps = await apply_depth_anything_v2(
                inputs["rgb_images"], **self.model_config
            )

            return NodeResult(
                status=NodeStatus.COMPLETED, data={"depth_maps": depth_maps}
            )
        except Exception as e:
            return NodeResult(status=NodeStatus.FAILED, error=e)
