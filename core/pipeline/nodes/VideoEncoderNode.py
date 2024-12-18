import cv2
from typing import Dict, List
from core.data_models import ImageData

from core.pipeline.base import Node, NodeResult, NodeStatus


class VideoEncoderNode(Node):
    """Node for encoding frames into a video file."""

    def __init__(
        self,
        output_path: str,
        filename: str,
        fps: int,
        codec: str = "mp4v",
    ):
        super().__init__(
            name="video_encoder",
            inputs={"frames", "file_name"},
            outputs={"video_path"},
        )
        self.output_path = output_path
        self.filename = filename
        self.fps = fps
        self.codec = codec

    async def execute(self, inputs: Dict[str, List[ImageData]]) -> NodeResult:
        try:
            frames = inputs["frames"]
            file_name = inputs["file_name"]

            if not frames:
                raise ValueError("No frames provided for video encoding")

            height, width = frames[0].image.shape[:2]
            output_file = f"{self.output_path}/{file_name}{self.filename}"

            writer = cv2.VideoWriter(
                output_file,
                cv2.VideoWriter_fourcc(*self.codec),
                self.fps,
                (width, height),
            )

            for frame_data in frames:
                writer.write(frame_data.image)

            writer.release()

            return NodeResult(
                status=NodeStatus.COMPLETED, data={"video_path": output_file}
            )
        except Exception as e:
            return NodeResult(status=NodeStatus.FAILED, error=e)
