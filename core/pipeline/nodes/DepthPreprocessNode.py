from typing import Dict, List
from core.data_models import ImageData

from core.pipeline.base import Node, NodeResult, NodeStatus


class DepthPreprocessNode(Node):
    """Node for preprocessing depth images."""

    def __init__(self):
        super().__init__(
            name="depth_preprocess", inputs={"depth_images"}, outputs={"depth_maps"}
        )

    async def execute(self, inputs: Dict[str, List[ImageData]]) -> NodeResult:
        try:
            depth_maps = []
            for depth_data in inputs["depth_images"]:
                # Just copy the depth data as-is
                depth_maps.append(
                    ImageData(
                        image=depth_data.image.copy(), timestamp=depth_data.timestamp
                    )
                )

            return NodeResult(
                status=NodeStatus.COMPLETED, data={"depth_maps": depth_maps}
            )
        except Exception as e:
            return NodeResult(status=NodeStatus.FAILED, error=e)
