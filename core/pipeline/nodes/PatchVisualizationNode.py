from typing import Dict, List
from core.data_models import ImageData

from core.tools.transform import (
    pose_to_transform_matrix,
    inverse_transform_matrix,
)
from core.tools.robots import (
    calculate_wheel_trajectories,
    robot_to_image,
    odom_to_robot,
)
from core.tools.data_processing import (
    find_closest_timestamp,
    filter_points_by_timestamp,
    filter_points_in_square,
    select_points,
)
from core.video.video_generation import (
    extract_and_visualize_patches_for_video,
)

from core.pipeline.base import Node, NodeResult, NodeStatus


class PatchVisualizationNode(Node):
    """Node for visualizing patches on frames."""

    def __init__(self, config: Dict):
        super().__init__(
            name="patch_visualization",
            inputs={"frames", "odometry_data", "robot_config"},
            outputs={"frames"},
        )
        self.video_config = config
        self.D = config.path_parameters.D
        self.d = config.path_parameters.d

    async def execute(self, inputs: Dict[str, List[ImageData]]) -> NodeResult:
        try:
            frames = inputs["frames"]
            odometry_data = inputs["odometry_data"]
            robot_config = inputs["robot_config"]
            frames_with_patches = []

            for frame_data in frames:
                frame = frame_data.image.copy()
                timestamp = frame_data.timestamp

                # Find closest odometry timestamp
                closest_timestamp = find_closest_timestamp(odometry_data, timestamp)
                if closest_timestamp is None:
                    frames_with_patches.append(frame_data)
                    continue

                # Filter odometry data
                filtered_odometry_data = filter_points_by_timestamp(
                    odometry_data, closest_timestamp
                )

                if (
                    filtered_odometry_data
                    and filtered_odometry_data[0]["movement_direction"] == "forward"
                ):
                    # Get transformations
                    ROBOT_TO_WORLD = pose_to_transform_matrix(filtered_odometry_data[0])
                    WORLD_TO_ROBOT = inverse_transform_matrix(ROBOT_TO_WORLD)
                    ROBOT_TO_CAM = inverse_transform_matrix(robot_config.cam_to_robot)

                    # Transform odometry to robot frame
                    odometry_data_into_robot = odom_to_robot(
                        filtered_odometry_data, WORLD_TO_ROBOT
                    )

                    # Filter points for patch extraction
                    filtered_odometry_data_into_robot = filter_points_in_square(
                        odometry_data_into_robot, self.D
                    )
                    filtered_odometry_data_into_robot = select_points(
                        filtered_odometry_data_into_robot, self.d
                    )

                    # Calculate wheel trajectories for patch bounds
                    left_wheel, right_wheel = calculate_wheel_trajectories(
                        filtered_odometry_data_into_robot,
                        wheel_distance=robot_config.wheel_distance,
                    )

                    # Transform to image coordinates
                    filtered_odometry_data_into_image = robot_to_image(
                        filtered_odometry_data_into_robot, ROBOT_TO_CAM, robot_config
                    )
                    left_wheel_data_into_image = robot_to_image(
                        left_wheel, ROBOT_TO_CAM, robot_config
                    )
                    right_wheel_data_into_image = robot_to_image(
                        right_wheel, ROBOT_TO_CAM, robot_config
                    )

                    # Extract and visualize patches
                    patches, frame = extract_and_visualize_patches_for_video(
                        frame,
                        filtered_odometry_data_into_image,
                        left_wheel_data_into_image,
                        right_wheel_data_into_image,
                    )

                frames_with_patches.append(ImageData(image=frame, timestamp=timestamp))

            return NodeResult(
                status=NodeStatus.COMPLETED,
                data={"frames": frames_with_patches},
            )
        except Exception as e:
            return NodeResult(status=NodeStatus.FAILED, error=e)
