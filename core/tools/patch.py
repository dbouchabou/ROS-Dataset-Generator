import os
import cv2
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def extract_patches(
    frame: np.ndarray,
    filtered_odometry_data_into_image: List[Dict[str, Any]],
    left_wheel_data_into_image: List[Dict[str, Any]],
    right_wheel_data_into_image: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract rectangular patches along the robot trajectory using image coordinates,
    and visualize them on the frame. Patches are extracted without perspective transformation.

    Args:
        frame (np.ndarray): The current video frame.
        filtered_odometry_data_into_image (List[Dict[str, Any]]): Filtered odometry data in image coordinates.
        left_wheel_data_into_image (List[Dict[str, Any]]): Left wheel trajectory in image coordinates.
        right_wheel_data_into_image (List[Dict[str, Any]]): Right wheel trajectory in image coordinates.

    Returns:
        List[Dict[str, Any]: List of extracted patches with metadata.
    """
    patches = []

    if len(filtered_odometry_data_into_image) < 2:
        print("Warning: Insufficient data points for patch extraction.")
        return patches

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

        except Exception as e:
            print(f"Error processing patch at index {i}: {str(e)}")
            continue

    return patches


def save_patches(
    patches: List[Dict[str, Any]], output_directory: str, base_filename: str = "patch"
):
    """
    Save extracted patches and their metadata to disk.

    Args:
        patches (List[Dict[str, Any]]): List of extracted patches with metadata.
        output_directory (str): Directory where patches and metadata will be saved.
        base_filename (str, optional): Base name for the saved files. Defaults to "patch".

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    for i, patch_data in enumerate(patches):
        # Generate filenames
        patch_filename = f"{base_filename}_{i:04d}.png"
        metadata_filename = f"{base_filename}_{i:04d}_metadata.json"

        # Save the patch image
        patch_path = os.path.join(output_directory, patch_filename)
        cv2.imwrite(patch_path, patch_data["patch"])

        # Prepare metadata (exclude the actual patch image from metadata)
        metadata = patch_data.copy()
        del metadata["patch"]

        # Convert numpy arrays to lists for JSON serialization
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()

        # Save metadata
        metadata_path = os.path.join(output_directory, metadata_filename)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # print(f"Saved {len(patches)} patches to {output_directory}")
