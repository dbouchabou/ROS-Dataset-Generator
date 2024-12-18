from typing import Dict, List
from core.data_models import ImageData

from core.tools.image import (
    add_text_to_frame,
)
from core.tools.data_processing import (
    find_closest_timestamp,
    filter_points_by_timestamp,
)

from core.pipeline.base import Node, NodeResult, NodeStatus


class AddInfoNode(Node):
    """Node for adds info on frame."""

    def __init__(self):
        super().__init__(
            name="add_info",
            inputs={"frames", "odometry_data"},
            outputs={"frames"},
        )

    async def execute(self, inputs: Dict[str, List[ImageData]]) -> NodeResult:
        try:
            frames = inputs["frames"]
            odometry_data = inputs["odometry_data"]
            frames_with_info = []

            for frame_data in frames:
                frame = frame_data.image.copy()
                timestamp = frame_data.timestamp

                # Find closest odometry timestamp
                closest_timestamp = find_closest_timestamp(odometry_data, timestamp)
                if closest_timestamp is None:
                    frames_with_info.append(frame_data)
                    continue

                # Filter odometry data
                filtered_odometry_data = filter_points_by_timestamp(
                    odometry_data, closest_timestamp
                )

                # Add motion information
                if filtered_odometry_data:
                    motion = filtered_odometry_data[0]
                    add_text_to_frame(
                        frame,
                        f"Motion: {motion['movement_direction']}",
                        position="bottom-left",
                    )

                    linear = motion["linear"]
                    add_text_to_frame(
                        frame,
                        f"Linear: {linear['x']:.2f}, {linear['y']:.2f}, {linear['z']:.2f}",
                        position="top-left",
                    )

                    angular = motion["angular"]
                    add_text_to_frame(
                        frame,
                        f"Angular: {angular['x']:.2f}, {angular['y']:.2f}, {angular['z']:.2f}",
                        position="top-right",
                    )

                frames_with_info.append(ImageData(image=frame, timestamp=timestamp))

            return NodeResult(
                status=NodeStatus.COMPLETED, data={"frames": frames_with_info}
            )
        except Exception as e:
            return NodeResult(status=NodeStatus.FAILED, error=e)
