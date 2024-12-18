import math
import bisect
import numpy as np


def filter_points_by_timestamp(points, reference_timestamp):
    """
    Filter points to keep only those with timestamps above the reference timestamp.

    Args:
        points (list): List of point dictionaries.
        reference_timestamp (float): Reference timestamp.

    Returns:
        list: Filtered list of points.
    """
    if reference_timestamp is None:
        print("Warning: reference_timestamp is None. Returning all points.")
        return points

    return [point for point in points if point["timestamp"] >= reference_timestamp]


def find_closest_timestamp(odometry_data, reference_timestamp):
    """
    Find the closest timestamp in odometry data to a reference timestamp.

    Args:
        odometry_data (list): List of odometry data dictionaries.
        reference_timestamp (float): Reference timestamp.

    Returns:
        float: Closest timestamp found in odometry data, or None if no suitable timestamp is found.
    """
    future_data = [
        data for data in odometry_data if data["timestamp"] >= reference_timestamp
    ]

    if not future_data:
        # print(f"Warning: No timestamps found after {reference_timestamp}")
        return None

    return min(future_data, key=lambda x: abs(x["timestamp"] - reference_timestamp))[
        "timestamp"
    ]


def filter_timestamps_intersection(odometry_data, odom_filter):
    """
    Filters odometry data to include only those entries that match the timestamps
    from a filtered set of odometry data.

    Args:
    odometry_data (list): List of odometry data points, where each point is a dictionary containing 'position' and 'timestamp'.
    odom_filter (list): List of filtered odometry data points, where each point is a dictionary containing 'position' and 'timestamp'.

    Returns:
    list: A list of odometry data points that match the timestamps from the filtered odometry data.
    """
    # Extract timestamps from the filtered odometry data
    filtered_timestamps = [entry["timestamp"] for entry in odom_filter]

    # Filter the odometry data based on these timestamps
    filtered_odometry_data = [
        entry for entry in odometry_data if entry["timestamp"] in filtered_timestamps
    ]

    return filtered_odometry_data


def filter_points_in_square(points, D):
    """
    Filters points that are within a square of side length 2*D centered at the reference position (first point),
    continuing to add points until a point goes outside this distance, and stopping if the robot moves backward.

    Args:
    points (list): List of points, where each point is a dictionary containing 'position' and 'timestamp'.
    D (float): The Euclidean distance defining the square's half-side length.

    Returns:
    list: A list of points within the specified square.
    """
    if not points:
        return []

    reference_position = points[0]["position"]
    filtered_points = []

    for point in points:

        # print(f"point: {point}")

        if point["movement_direction"] == "backward":
            break  # Stop adding points if the robot moves backward
            # continue

        distance = np.linalg.norm(
            [
                point["position"]["x"] - reference_position["x"],
                point["position"]["y"] - reference_position["y"],
            ]
        )

        if distance <= D:
            filtered_points.append(point)
        else:
            break  # Stop adding points once a point goes outside the specified distance

    return filtered_points


def calculate_distance(point1, point2):
    return math.sqrt(
        (point2["x"] - point1["x"]) ** 2 + (point2["y"] - point1["y"]) ** 2
    )


def calculate_angle(linear1, linear2):
    dot_product = linear1["x"] * linear2["x"] + linear1["y"] * linear2["y"]
    magnitude1 = math.sqrt(linear1["x"] ** 2 + linear1["y"] ** 2)
    magnitude2 = math.sqrt(linear2["x"] ** 2 + linear2["y"] ** 2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    cosine_angle = max(
        -1.0, min(1.0, cosine_angle)
    )  # Clamp the cosine value to avoid domain errors
    angle = math.acos(cosine_angle)
    return math.degrees(angle)


def select_points(odometry_data, d):

    if odometry_data == []:
        return odometry_data

    selected_points = []
    previous_point = odometry_data[0]["position"]
    previous_linear = odometry_data[0]["linear"]
    selected_points.append(odometry_data[0])

    for i in range(1, len(odometry_data)):
        current_point = odometry_data[i]["position"]
        current_linear = odometry_data[i]["linear"]
        distance = calculate_distance(previous_point, current_point)

        if distance >= d:
            angle = calculate_angle(previous_linear, current_linear)
            if angle >= 45 or angle < -45:
                break

            selected_points.append(odometry_data[i])
            previous_point = current_point
            previous_linear = current_linear

    return selected_points


def find_closest_cmd_vel(timestamp, cmd_vel_data):
    """Find the cmd_vel data with the closest timestamp."""
    timestamps = [cmd["timestamp"] for cmd in cmd_vel_data]
    index = bisect.bisect_left(timestamps, timestamp)
    if index == 0:
        return cmd_vel_data[0]
    if index == len(cmd_vel_data):
        return cmd_vel_data[-1]
    before = cmd_vel_data[index - 1]
    after = cmd_vel_data[index]
    if after["timestamp"] - timestamp < timestamp - before["timestamp"]:
        return after
    else:
        return before
