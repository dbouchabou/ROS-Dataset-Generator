# video/nodes.py

import cv2
import numpy as np
from typing import Dict, List, Optional

from data_models import ImageData
from tools.image import (
    draw_points_on_frame,
    add_text_to_frame,
    convert_depth_images_to_normal_images,
    draw_polygon_on_frame,
    is_within_image,
)

from tools.transform import (
    pose_to_transform_matrix,
    inverse_transform_matrix,
    apply_rigid_motion,
    camera_frame_to_image,
)
from tools.robots import (
    calculate_wheel_trajectories,
    odom_to_image,
    robot_to_image,
    odom_to_robot,
)
from data_processing import (
    find_closest_timestamp,
    filter_points_by_timestamp,
    filter_timestamps_intersection,
    filter_points_in_square,
    select_points,
)

from .video_generation import (
    extract_and_visualize_patches_for_video,
)

from pipeline.base import Node, NodeResult, NodeStatus
