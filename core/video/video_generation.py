# video/video_generation.py

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Tuple, Optional

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.console import Console
from core.data_models import DatasetConfig, ImageData
from core.tools.image import (
    is_within_image,
    add_text_to_frame,
    draw_points_on_frame,
    draw_rectangle_on_frame,
    draw_polygon_on_frame,
)
from core.tools.depth import (
    depth_to_grayscale,
    visualize_depth_map,
    grayscale_to_3channel,
)
from core.tools.patch import save_patches
from core.tools.robots import (
    calculate_wheel_trajectories,
    odom_to_image,
    robot_to_image,
    odom_to_robot,
)
from core.tools.data_processing import (
    find_closest_timestamp,
    filter_points_by_timestamp,
    filter_timestamps_intersection,
    filter_points_in_square,
    select_points,
)
from core.tools.transform import (
    pose_to_transform_matrix,
    inverse_transform_matrix,
    apply_rigid_motion,
    camera_frame_to_image,
)


def extract_and_visualize_patches_for_video(
    frame: np.ndarray,
    filtered_odometry_data_into_image: List[Dict[str, Any]],
    left_wheel_data_into_image: List[Dict[str, Any]],
    right_wheel_data_into_image: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Extract rectangular patches along the robot trajectory using image coordinates,
    and visualize them on the frame. Patches are extracted without perspective transformation.

    Args:
        frame (np.ndarray): The current video frame.
        filtered_odometry_data_into_image (List[Dict[str, Any]]): Filtered odometry data in image coordinates.
        left_wheel_data_into_image (List[Dict[str, Any]]): Left wheel trajectory in image coordinates.
        right_wheel_data_into_image (List[Dict[str, Any]]): Right wheel trajectory in image coordinates.

    Returns:
        Tuple[List[Dict[str, Any]], np.ndarray]: List of extracted patches with metadata and the modified frame.
    """
    patches = []

    if len(filtered_odometry_data_into_image) < 2:
        # print("Warning: Insufficient data points for patch extraction.")
        return patches, frame

    for i in range(len(filtered_odometry_data_into_image) - 1):
        try:
            left_current = np.array(
                [
                    left_wheel_data_into_image[i]["position"]["x"],
                    left_wheel_data_into_image[i]["position"]["y"],
                ]
            )
            right_current = np.array(
                [
                    right_wheel_data_into_image[i]["position"]["x"],
                    right_wheel_data_into_image[i]["position"]["y"],
                ]
            )
            left_next = np.array(
                [
                    left_wheel_data_into_image[i + 1]["position"]["x"],
                    left_wheel_data_into_image[i + 1]["position"]["y"],
                ]
            )
            right_next = np.array(
                [
                    right_wheel_data_into_image[i + 1]["position"]["x"],
                    right_wheel_data_into_image[i + 1]["position"]["y"],
                ]
            )

            # Calculate patch corners
            top_left = left_current
            top_right = right_current
            bottom_left = left_next
            bottom_right = right_next

            corners_image = np.array([top_left, top_right, bottom_right, bottom_left])

            # Check if the patch is within the frame
            if all(
                0 <= corner[0] < frame.shape[1] and 0 <= corner[1] < frame.shape[0]
                for corner in corners_image
            ):
                # Extract patch by cropping
                x_min = int(min(corner[0] for corner in corners_image))
                y_min = int(min(corner[1] for corner in corners_image))
                x_max = int(max(corner[0] for corner in corners_image))
                y_max = int(max(corner[1] for corner in corners_image))

                patch = frame[y_min:y_max, x_min:x_max]

                patches.append(
                    {
                        "patch": patch,
                        "timestamp_start": filtered_odometry_data_into_image[i][
                            "timestamp"
                        ],
                        "timestamp_end": filtered_odometry_data_into_image[i + 1][
                            "timestamp"
                        ],
                        "position_start": filtered_odometry_data_into_image[i][
                            "position"
                        ],
                        "position_end": filtered_odometry_data_into_image[i + 1][
                            "position"
                        ],
                        "corners_image": corners_image.tolist(),
                        "movement_direction": filtered_odometry_data_into_image[i].get(
                            "movement_direction", "unknown"
                        ),
                    }
                )

                # Draw quadrilateral on frame
                frame = draw_polygon_on_frame(
                    frame,
                    corners_image.tolist(),
                    color=(0, 255, 0),  # Green color
                    thickness=2,
                )

        except Exception as e:
            print(f"Error processing patch at index {i}: {str(e)}")
            continue

    return patches, frame


##########


# def process_frame(
#     args: [
#         ImageData,
#         List[Dict[str, Any]],
#         DatasetConfig,
#         float,
#         float,
#         bool,
#         str,
#         bool,
#         bool,
#         int,
#     ]
# ) -> np.ndarray:
def process_frame(
    image_data: ImageData,
    frame_index: int,
    odometry_data: List[Dict[str, Any]],
    config: DatasetConfig,
    D: float,
    d: float,
    depth_image: bool,
    depth_colormap: str,
    draw_patches: bool,
    draw_trajectories: bool,
) -> Any:
    """
    Process a single frame for video generation.

    Args:
        args (Tuple): A tuple containing:
            - image_data (Dict[str, Any]): Image data dictionary.
            - odometry_data (List[Dict[str, Any]]): List of odometry data dictionaries.
            - config (DatasetConfig): Dataset configuration object.
            - D (float): Distance parameter for filtering points in square.
            - d (float): Distance threshold for selecting points.
            - frame_number (int): The current frame number.

    Returns:
        np.ndarray: Processed frame with projected points.
    """

    # (
    #     image_data,
    #     odometry_data,
    #     config,
    #     D,
    #     d,
    #     depth_image,
    #     depth_colormap,
    #     draw_patches,
    #     draw_trajectories,
    #     frame_number,
    # ) = args

    if depth_image:
        if depth_colormap == "grayscale":
            image = grayscale_to_3channel(depth_to_grayscale(image_data.image))
        else:
            image = visualize_depth_map(image_data.image, colormap=depth_colormap)
    else:
        image = image_data.image

    timestamp = image_data.timestamp

    frame = np.array(image).copy()
    closest_timestamp = find_closest_timestamp(odometry_data, timestamp)

    if closest_timestamp is None:
        return frame

    filtered_odometry_data = filter_points_by_timestamp(
        odometry_data, closest_timestamp
    )

    if not filtered_odometry_data:
        return frame

    if filtered_odometry_data[0]["movement_direction"] == "forward":
        ROBOT_TO_WORLD = pose_to_transform_matrix(filtered_odometry_data[0])
        WORLD_TO_ROBOT = inverse_transform_matrix(ROBOT_TO_WORLD)
        ROBOT_TO_CAM = inverse_transform_matrix(config.cam_to_robot)

        odom_into_image = odom_to_image(
            filtered_odometry_data, WORLD_TO_ROBOT, ROBOT_TO_CAM, config.K
        )
        filtered_odom_into_image = [
            point
            for point in odom_into_image
            if is_within_image(
                (point["position"]["x"], point["position"]["y"]),
                frame.shape[0],
                frame.shape[1],
            )
        ]

        filtered_3D_odometry_data = filter_timestamps_intersection(
            filtered_odometry_data, filtered_odom_into_image
        )
        odometry_data_into_robot = odom_to_robot(
            filtered_3D_odometry_data, WORLD_TO_ROBOT
        )
        filtered_odometry_data_into_robot = filter_points_in_square(
            odometry_data_into_robot, D
        )
        filtered_odometry_data_into_robot = select_points(
            filtered_odometry_data_into_robot, d
        )

        filtered_odometry_data_into_image = robot_to_image(
            filtered_odometry_data_into_robot, ROBOT_TO_CAM, config
        )

        # Compute wheel trajectories if patches or trajectories are to be drawn
        if draw_patches or draw_trajectories:
            left_wheel, right_wheel = calculate_wheel_trajectories(
                filtered_odometry_data_into_robot, wheel_distance=config.wheel_distance
            )

            left_wheel_data_into_image = robot_to_image(
                left_wheel, ROBOT_TO_CAM, config
            )
            right_wheel_data_into_image = robot_to_image(
                right_wheel, ROBOT_TO_CAM, config
            )

        # Draw trajectories on the frame
        if draw_trajectories:
            draw_trajectories_on_frame(
                frame,
                filtered_odometry_data_into_image,
                left_wheel_data_into_image,
                right_wheel_data_into_image,
            )

        # Extract patches and draw them on the frame
        if draw_patches:
            # Extract patches and draw them on the frame
            patches, frame = extract_and_visualize_patches_for_video(
                frame,
                filtered_odometry_data_into_image,
                left_wheel_data_into_image,
                right_wheel_data_into_image,
            )

        # Extract patches and draw them on the frame
        # patches, frame = extract_and_visualize_patches_for_video_2(
        #     frame,
        #     filtered_odometry_data_into_robot,
        #     ROBOT_TO_CAM,
        #     config.K,
        #     wheel_width=0.65,
        # )

        # Save patches
        # save_patches(patches, "output/patches", f"frame_{frame_number:04d}")

    add_frame_info(frame, filtered_odometry_data, timestamp, closest_timestamp)

    return frame


def draw_trajectories_on_frame(
    frame: np.ndarray,
    center_traj: List[Dict[str, Any]],
    left_traj: List[Dict[str, Any]],
    right_traj: List[Dict[str, Any]],
) -> None:
    """
    Draw trajectories on the frame.

    Args:
        frame (np.ndarray): The input frame.
        center_traj (List[Dict[str, Any]]): Center trajectory points.
        left_traj (List[Dict[str, Any]]): Left wheel trajectory points.
        right_traj (List[Dict[str, Any]]): Right wheel trajectory points.
    """
    center_points = [
        (int(point["position"]["x"]), int(point["position"]["y"]))
        for point in center_traj
    ]
    left_points = [
        (int(point["position"]["x"]), int(point["position"]["y"]))
        for point in left_traj
    ]
    right_points = [
        (int(point["position"]["x"]), int(point["position"]["y"]))
        for point in right_traj
    ]

    draw_points_on_frame(
        frame, center_points, radius=5, color=(0, 255, 0), thickness=-1
    )
    draw_points_on_frame(frame, left_points, radius=5, color=(255, 0, 0), thickness=-1)
    draw_points_on_frame(frame, right_points, radius=5, color=(0, 0, 255), thickness=-1)


def add_frame_info(
    frame: np.ndarray,
    odometry_data: List[Dict[str, Any]],
    timestamp: float,
    closest_timestamp: float,
) -> None:
    """
    Add information to the frame.

    Args:
        frame (np.ndarray): The input frame.
        odometry_data (List[Dict[str, Any]]): Odometry data.
        timestamp (float): Current timestamp.
        closest_timestamp (float): Closest odometry timestamp.
    """
    if odometry_data:
        add_text_to_frame(
            frame,
            f"Motion: {odometry_data[0]['movement_direction']}",
            position="bottom-left",
        )
        linear = odometry_data[0]["linear"]
        angular = odometry_data[0]["angular"]
        add_text_to_frame(
            frame,
            f"Linear: {linear['x']:.2f}, {linear['y']:.2f}, {linear['z']:.2f}",
            position="top-left",
        )
        add_text_to_frame(
            frame,
            f"Angular: {angular['x']:.2f}, {angular['y']:.2f}, {angular['z']:.2f}",
            position="top-right",
        )

    time_diff = timestamp - closest_timestamp
    add_text_to_frame(frame, f"Time diff: {time_diff:.3f}s")


def generate_video(
    image_data_list: List[ImageData],
    odometry_data: List[Dict[str, Any]],
    config: DatasetConfig,
    output_path: str,
    video_filename: str,
    frame_rate: int,
    D: float = 1.0,
    d: float = 0.1,
    depth_image: bool = False,
    depth_colormap: str = "grayscale",
    num_processes: int = None,
    draw_patches: bool = False,
    draw_trajectories: bool = False,
) -> None:
    """
    Generate a video with projected odometry points.

    Args:
        image_data_list (List[ImageData]): List of ImageData objects containing images and timestamps.
        odometry_data (List[Dict[str, Any]]): List of odometry data dictionaries.
        config (DatasetConfig): Dataset configuration object.
        output_path (str): Output directory path.
        video_filename (str): Name of the output video file.
        frame_rate (int): Frame rate of the output video.
        D (float, optional): Distance parameter for filtering points in square. Defaults to 1.0.
        d (float, optional): Distance threshold for selecting points. Defaults to 0.1.
        depth_image (bool, optional): If True, uses depth images. Defaults to False.
        depth_colormap (str, optional): Colormap to use for depth images. Defaults to "grayscale".
        num_processes (int, optional): Number of processes to use. If None, it will use the number of CPU cores.
        draw_patches (bool, optional): If True, draws patches on the frames. Defaults to False.
        draw_trajectories (bool, optional): If True, draws trajectories on the frames. Defaults to False.

    Returns:
        None
    """
    # Create the output directory if it does not exist
    os.makedirs(output_path, exist_ok=True)

    # Number of processes to use. If None, use the number of CPU cores
    if num_processes is None:
        num_processes = cpu_count()

    # Get the dimensions of the first image
    height, width = image_data_list[0].image.shape[:2]

    # OpenCV expects (width, height) for video size
    video_size = (width, height)

    console = Console()
    console.print(f"Video size: {video_size}")
    console.print(f"Frame rate: {frame_rate} fps")
    console.print(f"Total frames: {len(image_data_list)}")

    # Initialize the video writer
    video_writer = cv2.VideoWriter(
        os.path.join(output_path, video_filename),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        video_size,
    )

    # Prepare the arguments for processing frames
    args_list = [
        (
            image_data,
            i,
        )
        for i, image_data in enumerate(image_data_list)
    ]

    # Create a partial function to fix common arguments
    process_frame_partial = partial(
        process_frame,
        odometry_data=odometry_data,
        config=config,
        D=D,
        d=d,
        depth_image=depth_image,
        depth_colormap=depth_colormap,
        draw_patches=draw_patches,
        draw_trajectories=draw_trajectories,
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style="yellow"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        total_frames = len(args_list)
        # Combined progress task for processing and writing frames
        progress_task = progress.add_task(
            "[cyan]Processing and writing frames", total=total_frames
        )

        # Use multiprocessing Pool to process frames in parallel
        with Pool(processes=num_processes) as pool:
            for frame in pool.starmap(process_frame_partial, args_list):
                video_writer.write(frame)
                progress.update(progress_task, advance=1)

    # Release the video writer
    video_writer.release()

    console.print(
        f"[bold green]Voila!✨ Video saved as 👉 {video_filename} in the folder {output_path}[/bold green]"
    )
