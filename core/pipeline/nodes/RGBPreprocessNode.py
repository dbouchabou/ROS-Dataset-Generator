import numpy as np
from typing import Dict, List
from core.data_models import ImageData

from core.pipeline.base import Node, NodeResult, NodeStatus


class RGBPreprocessNode(Node):
    """Node for preprocessing RGB images."""

    def __init__(self):
        super().__init__(
            name="rgb_preprocess",
            inputs={"rgb_images"},
            outputs={"frames"},
        )

    async def execute(self, inputs: Dict[str, List[ImageData]]) -> NodeResult:
        try:
            frames = []
            for img_data in inputs["rgb_images"]:
                frame = np.array(img_data.image).copy()
                frames.append(ImageData(image=frame, timestamp=img_data.timestamp))

            return NodeResult(status=NodeStatus.COMPLETED, data={"frames": frames})
        except Exception as e:
            return NodeResult(status=NodeStatus.FAILED, error=e)
