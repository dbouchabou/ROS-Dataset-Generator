from typing import List, Dict, Any

import numpy as np
import rosbag
from cv_bridge import CvBridge
from core.data_models import ImageData
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from core.tools.depth import depth_to_grayscale


def read_rgb_image_data(bag_file: str, topic: str) -> List[ImageData]:
    """
    Read RGB image data from a ROS bag file.

    Args:
        bag_file (str): Path to the ROS bag file.
        topic (str): Topic name for image data.

    Returns:
        list: List of image data dictionaries.
        ImageDataList: List of ImageData objects.
    """
    bag = rosbag.Bag(bag_file)
    # image_msgs = []
    image_data_list = []
    bridge = CvBridge()

    total_messages = bag.get_message_count(topic_filters=[topic])
    console = Console()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style="yellow"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]👁️  Reading {topic}", total=total_messages)

        for _, msg, t in bag.read_messages(topics=[topic]):
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                # image_data = {"image": cv_image, "timestamp": t.to_sec()}
                # image_msgs.append(image_data)
                image_data = ImageData(image=cv_image, timestamp=t.to_sec())
                image_data_list.append(image_data)
            except Exception as e:
                console.print(f"[bold red]Error converting image:[/bold red] {e}")
            progress.update(task, advance=1)

    bag.close()

    console.print(
        # f"[bold green]Successfully converted {len(image_msgs)} out of {total_messages} images from topic {topic}[/bold green]"
        f"[bold green]Successfully converted {len(image_data_list)} out of {total_messages} images from topic {topic}[/bold green]"
    )

    # return image_msgs
    return image_data_list


def read_depth_image_data(bag_file: str, topic: str) -> List[ImageData]:
    """
    Read depth image data from a ROS bag file and convert to grayscale.

    Args:
        bag_file (str): Path to the ROS bag file.
        topic (str): Topic name for depth image data.

    Returns:
        List[ImageData]: List of ImageData objects containing raw depth images.
    """
    bag = rosbag.Bag(bag_file)
    image_data_list = []
    bridge = CvBridge()

    total_messages = bag.get_message_count(topic_filters=[topic])
    console = Console()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style="yellow"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]👁️  Reading depth data from {topic}", total=total_messages
        )

        for _, msg, t in bag.read_messages(topics=[topic]):
            try:
                # Convert depth image using passthrough encoding
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

                # Convert depth image to grayscale
                # grayscale_image = depth_to_grayscale(cv_image)

                image_data = ImageData(image=cv_image, timestamp=t.to_sec())
                image_data_list.append(image_data)
            except Exception as e:
                console.print(f"[bold red]Error converting depth image:[/bold red] {e}")
            progress.update(task, advance=1)

    bag.close()

    console.print(
        f"[bold green]Successfully converted {len(image_data_list)} out of {total_messages} depth images from topic {topic}[/bold green]"
    )

    return image_data_list


def read_odometry_data(bag_file, topic):
    """
    Read odometry data from a ROS bag file with Rich progress bar.

    Args:
        bag_file (str): Path to the ROS bag file.
        topic (str): Topic name for odometry data.

    Returns:
        list: List of odometry data dictionaries.
    """
    bag = rosbag.Bag(bag_file)
    odometry_msgs = []

    total_messages = bag.get_message_count(topic_filters=[topic])
    console = Console()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style="yellow"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]👁️  Reading odometry from {topic}", total=total_messages
        )

        for _, msg, t in bag.read_messages(topics=[topic]):
            odometry_data = {
                "position": {
                    "x": msg.pose.pose.position.x,
                    "y": msg.pose.pose.position.y,
                    # "z": msg.pose.pose.position.z,
                    "z": 0,
                },
                "orientation": {
                    "x": msg.pose.pose.orientation.x,
                    "y": msg.pose.pose.orientation.y,
                    "z": msg.pose.pose.orientation.z,
                    "w": msg.pose.pose.orientation.w,
                },
                "linear": {
                    "x": msg.twist.twist.linear.x,
                    "y": msg.twist.twist.linear.y,
                    "z": msg.twist.twist.linear.z,
                },
                "angular": {
                    "x": msg.twist.twist.angular.x,
                    "y": msg.twist.twist.angular.y,
                    "z": msg.twist.twist.angular.z,
                },
                "timestamp": t.to_sec(),
            }
            odometry_msgs.append(odometry_data)
            progress.update(task, advance=1)

    bag.close()
    console.print(
        f"[bold green]Read {len(odometry_msgs)} odometry messages from topic {topic}[/bold green]"
    )
    return odometry_msgs


def read_cmd_vel_data(bag_file, topic):
    """
    Read cmd_vel data from a ROS bag file with Rich progress bar.
    Handles both TwistStamped and Twist message types.

    Args:
        bag_file (str): Path to the ROS bag file.
        topic (str): Topic name for cmd_vel data.

    Returns:
        list: List of cmd_vel data dictionaries.
    """
    bag = rosbag.Bag(bag_file)
    cmd_vel_msgs = []

    total_messages = bag.get_message_count(topic_filters=[topic])
    console = Console()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style="yellow"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]👁️  Reading {topic}", total=total_messages)

        for _, msg, t in bag.read_messages(topics=[topic]):
            # Check if message has header (TwistStamped)
            if hasattr(msg, "header"):
                twist = msg.twist
                # timestamp = msg.header.stamp.to_sec()
            else:
                # Regular Twist message
                twist = msg

            timestamp = t.to_sec()

            cmd_vel_data = {
                "linear": {
                    "x": twist.linear.x,
                    "y": twist.linear.y,
                    "z": twist.linear.z,
                },
                "angular": {
                    "x": twist.angular.x,
                    "y": twist.angular.y,
                    "z": twist.angular.z,
                },
                "timestamp": timestamp,
            }

            cmd_vel_msgs.append(cmd_vel_data)
            progress.update(task, advance=1)

    bag.close()
    console.print(
        f"[bold green]Read {len(cmd_vel_msgs)} cmd_vel messages from topic {topic}[/bold green]"
    )
    # console.print(f"[bold green]Example {cmd_vel_msgs[100]}[/bold green]")
    return cmd_vel_msgs


def read_imu_data(bag_file: str, topic: str) -> List[Dict[str, Any]]:
    """
    Read IMU data from a ROS bag file.

    Args:
        bag_file (str): Path to the ROS bag file.
        topic (str): Topic name for IMU data.

    Returns:
        List[Dict[str, Any]]: List of IMU data dictionaries.
    """
    bag = rosbag.Bag(bag_file)
    imu_msgs = []

    total_messages = bag.get_message_count(topic_filters=[topic])
    console = Console()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style="yellow"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]👁️  Reading IMU data from {topic}", total=total_messages
        )

        for _, msg, t in bag.read_messages(topics=[topic]):
            imu_data = {
                "timestamp": t.to_sec(),
                "orientation": {
                    "x": msg.orientation.x,
                    "y": msg.orientation.y,
                    "z": msg.orientation.z,
                    "w": msg.orientation.w,
                },
                "angular_velocity": {
                    "x": msg.angular_velocity.x,
                    "y": msg.angular_velocity.y,
                    "z": msg.angular_velocity.z,
                },
                "linear_acceleration": {
                    "x": msg.linear_acceleration.x,
                    "y": msg.linear_acceleration.y,
                    "z": msg.linear_acceleration.z,
                },
                "orientation_covariance": msg.orientation_covariance,
                "angular_velocity_covariance": msg.angular_velocity_covariance,
                "linear_acceleration_covariance": msg.linear_acceleration_covariance,
            }
            imu_msgs.append(imu_data)
            progress.update(task, advance=1)

    bag.close()
    console.print(
        f"[bold green]Read {len(imu_msgs)} IMU messages from topic {topic}[/bold green]"
    )
    return imu_msgs


def read_motor_rpm(bag_file: str, topic: str) -> List[Dict[str, Any]]:
    """
    Read motor RPM data from a ROS bag file.

    Args:
        bag_file (str): Path to the ROS bag file.
        topic (str): Topic name for motor RPM data.

    Returns:
        List[Dict[str, Any]]: List of motor RPM data dictionaries.
    """
    bag = rosbag.Bag(bag_file)
    rpm_msgs = []
    total_messages = bag.get_message_count(topic_filters=[topic])
    console = Console()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style="yellow"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]🔄 Reading Motor RPM data from {topic}", total=total_messages
        )

        for _, msg, t in bag.read_messages(topics=[topic]):
            rpm_data = {
                "timestamp": t.to_sec(),
                "motor_values": list(msg.data),
                "layout": {
                    "dim": msg.layout.dim,
                    "data_offset": msg.layout.data_offset,
                },
            }
            rpm_msgs.append(rpm_data)
            progress.update(task, advance=1)

    bag.close()
    console.print(
        f"[bold green]Read {len(rpm_msgs)} Motor RPM messages from topic {topic}[/bold green]"
    )
    return rpm_msgs


def estimate_frame_rate(bag_file, image_topic):
    """
    Estimate the frame rate of images in a ROS bag file.

    Args:
        bag_file (str): Path to the ROS bag file.
        image_topic (str): The topic name for image messages.

    Returns:
        float: Estimated frame rate in frames per second.
    """
    from collections import deque

    import rosbag

    bag = rosbag.Bag(bag_file)
    timestamps = deque(maxlen=100)  # Store last 100 timestamps

    for _, _, t in bag.read_messages(topics=[image_topic]):
        timestamps.append(t.to_sec())

        if len(timestamps) == 100:
            break

    bag.close()

    if len(timestamps) < 2:
        return None  # Not enough messages to estimate frame rate

    # Calculate average time difference between frames
    time_diffs = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    avg_time_diff = sum(time_diffs) / len(time_diffs)

    # Frame rate is the inverse of the average time difference
    frame_rate = 1 / avg_time_diff

    return frame_rate
