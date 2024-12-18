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
from core.tools.image import draw_points_on_frame

from core.pipeline.base import Node, NodeResult, NodeStatus


class TrajectoryVisualizationNode(Node):
    """Node for visualizing robot trajectories on frames."""

    def __init__(self, config: Dict):
        super().__init__(
            name="trajectory_visualization",
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
            frames_with_trajectory = []

            for frame_data in frames:
                frame = frame_data.image.copy()
                timestamp = frame_data.timestamp

                # Find closest odometry timestamp
                closest_timestamp = find_closest_timestamp(odometry_data, timestamp)
                if closest_timestamp is None:
                    frames_with_trajectory.append(frame_data)
                    continue

                # Filter odometry data
                filtered_odometry_data = filter_points_by_timestamp(
                    odometry_data, closest_timestamp
                )

                if (
                    filtered_odometry_data
                    and filtered_odometry_data[0]["movement_direction"] == "forward"
                ):
                    # Calculate transformations
                    ROBOT_TO_WORLD = pose_to_transform_matrix(filtered_odometry_data[0])
                    WORLD_TO_ROBOT = inverse_transform_matrix(ROBOT_TO_WORLD)
                    ROBOT_TO_CAM = inverse_transform_matrix(robot_config.cam_to_robot)

                    # Transform to robot frame
                    odometry_data_into_robot = odom_to_robot(
                        filtered_odometry_data, WORLD_TO_ROBOT
                    )

                    # Filter points for trajectory
                    filtered_odometry_data_into_robot = filter_points_in_square(
                        odometry_data_into_robot, self.D
                    )
                    filtered_odometry_data_into_robot = select_points(
                        filtered_odometry_data_into_robot, self.d
                    )

                    # Calculate wheel trajectories
                    left_wheel, right_wheel = calculate_wheel_trajectories(
                        filtered_odometry_data_into_robot,
                        wheel_distance=robot_config.wheel_distance,
                    )

                    # Transform to image coordinates
                    center_trajectory = robot_to_image(
                        filtered_odometry_data_into_robot, ROBOT_TO_CAM, robot_config
                    )
                    left_trajectory = robot_to_image(
                        left_wheel, ROBOT_TO_CAM, robot_config
                    )
                    right_trajectory = robot_to_image(
                        right_wheel, ROBOT_TO_CAM, robot_config
                    )

                    # Draw trajectories
                    frame = draw_points_on_frame(
                        frame,
                        [
                            (int(p["position"]["x"]), int(p["position"]["y"]))
                            for p in center_trajectory
                        ],
                        radius=5,
                        color=(0, 255, 0),
                        thickness=-1,
                    )
                    frame = draw_points_on_frame(
                        frame,
                        [
                            (int(p["position"]["x"]), int(p["position"]["y"]))
                            for p in left_trajectory
                        ],
                        radius=5,
                        color=(255, 0, 0),
                        thickness=-1,
                    )
                    frame = draw_points_on_frame(
                        frame,
                        [
                            (int(p["position"]["x"]), int(p["position"]["y"]))
                            for p in right_trajectory
                        ],
                        radius=5,
                        color=(0, 0, 255),
                        thickness=-1,
                    )

                frames_with_trajectory.append(
                    ImageData(image=frame, timestamp=timestamp)
                )

            return NodeResult(
                status=NodeStatus.COMPLETED,
                data={"frames": frames_with_trajectory},
            )
        except Exception as e:
            return NodeResult(status=NodeStatus.FAILED, error=e)
