import cv2
import warnings
import numpy as np
from typing import List, Tuple, Union
from multiprocessing import Pool, cpu_count

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.console import Console

# Assuming ImageData is defined in your data_models module
from core.data_models import ImageData


def is_within_image(
    point: Tuple[int, int], image_height: int, image_width: int
) -> bool:
    """
    Check if a point is within the image boundaries.

    Args:
        point (Tuple[int, int]): (x, y) coordinates of the point.
        image_height (int): Height of the image.
        image_width (int): Width of the image.

    Returns:
        bool: True if the point is within the image, False otherwise.
    """
    x, y = point
    return 0 <= x < image_width and 0 <= y < image_height


def draw_points_on_frame(
    frame: np.ndarray,
    points_list: List[Tuple[int, int]],
    radius: int = 5,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = -1,
) -> np.ndarray:
    """
    Draw points on the frame from the filtered odometry data.

    Args:
        frame (np.ndarray): The input frame.
        points_list (List[Tuple[int, int]]): List of (x, y) coordinates.
        radius (int): Radius of the circle to draw. Default is 5.
        color (Tuple[int, int, int]): Color of the circle in BGR format. Default is green (0, 255, 0).
        thickness (int): Thickness of the circle border. Default is -1 (filled circle).

    Returns:
        np.ndarray: Frame with points drawn on it.
    """
    for point in points_list:
        x, y = point
        if is_within_image((x, y), frame.shape[0], frame.shape[1]):
            cv2.circle(frame, (x, y), radius=radius, color=color, thickness=thickness)
    return frame


def draw_rectangle_on_frame(
    frame: np.ndarray,
    top_left: Tuple[float, float],
    bottom_right: Tuple[float, float],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a rectangle on the frame using the top-left and bottom-right corners.

    Args:
        frame (np.ndarray): The input frame.
        top_left (Tuple[float, float]): (x, y) coordinates of the top-left corner.
        bottom_right (Tuple[float, float]): (x, y) coordinates of the bottom-right corner.
        color (Tuple[int, int, int]): Color of the rectangle in BGR format. Default is green (0, 255, 0).
        thickness (int): Thickness of the rectangle border. Default is 2.

    Returns:
        np.ndarray: Frame with the rectangle drawn on it.
    """
    cv2.rectangle(
        frame,
        tuple(map(int, top_left)),
        tuple(map(int, bottom_right)),
        color,
        thickness,
    )
    return frame


def draw_polygon_on_frame(
    frame: np.ndarray,
    corners: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a polygon on the frame using the given corners.

    Args:
        frame (np.ndarray): The input frame.
        corners (List[Tuple[float, float]]): List of (x, y) tuples representing the corners.
        color (Tuple[int, int, int]): Color of the polygon in BGR format. Default is green (0, 255, 0).
        thickness (int): Thickness of the polygon border. Default is 2.

    Returns:
        np.ndarray: Frame with the polygon drawn on it.
    """
    points = np.array(corners, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=thickness)
    return frame


def add_text_to_frame(
    frame: np.ndarray,
    text: str,
    position: Union[str, Tuple[int, int]] = "center",
    color: Tuple[int, int, int] = (0, 0, 255),
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    font_thickness: int = 2,
) -> np.ndarray:
    """
    Add text to the frame at a specified position with a specified color.

    Args:
        frame (np.ndarray): The input frame.
        text (str): The text to display.
        position (Union[str, Tuple[int, int]]): The position of the text. Can be 'center', 'top-left', 'top-right',
            'bottom-left', 'bottom-right', or a tuple of (x, y) coordinates.
        color (Tuple[int, int, int]): The color of the text in BGR format. Default is red (0, 0, 255).
        font (int): Font type. Default is cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float): Font scale (size). Default is 1.0.
        font_thickness (int): Thickness of the font. Default is 2.

    Returns:
        np.ndarray: Frame with added text.
    """
    height, width = frame.shape[:2]

    # Get the size of the text
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Calculate the position of the text
    if position == "center":
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
    elif position == "top-left":
        text_x = 10
        text_y = text_size[1] + 10
    elif position == "top-right":
        text_x = width - text_size[0] - 10
        text_y = text_size[1] + 10
    elif position == "bottom-left":
        text_x = 10
        text_y = height - 10
    elif position == "bottom-right":
        text_x = width - text_size[0] - 10
        text_y = height - 10
    elif isinstance(position, tuple) and len(position) == 2:
        text_x, text_y = position
    else:
        raise ValueError(
            "Invalid position. Use 'center', 'top-left', 'top-right', 'bottom-left', "
            "'bottom-right', or a tuple of (x, y) coordinates."
        )

    # Add the text to the frame
    cv2.putText(
        frame,
        text,
        (int(text_x), int(text_y)),
        font,
        font_scale,
        color,
        font_thickness,
    )
    return frame


def compute_normals(depth_image: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """
    Compute normal vectors from a depth image using np.gradient, handling invalid values robustly.

    Args:
        depth_image (np.ndarray): Input depth image (single-channel, float32).
        scale_factor (float): Factor to scale the depth values. Defaults to 1.0.

    Returns:
        np.ndarray: 3-channel image of normal vectors (uint8).
    """
    # Ensure the input is a 2D array
    if depth_image.ndim != 2:
        raise ValueError("Input must be a single-channel (2D) depth image")

    # Apply scale factor and handle invalid values
    scaled_depth = np.where(
        np.isfinite(depth_image), depth_image * scale_factor, np.nan
    )

    # Compute gradients using np.gradient
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        dz_dv, dz_du = np.gradient(scaled_depth)

    # Replace NaN and inf values with zeros
    dz_dv = np.nan_to_num(dz_dv, nan=0.0, posinf=0.0, neginf=0.0)
    dz_du = np.nan_to_num(dz_du, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute normal components
    Nx = -dz_du
    Ny = -dz_dv
    Nz = np.ones_like(scaled_depth)

    # Normalize the normal vectors
    norm = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    norm = np.where(norm == 0, np.finfo(float).eps, norm)

    Nx = np.divide(Nx, norm, out=np.zeros_like(Nx), where=norm != 0)
    Ny = np.divide(Ny, norm, out=np.zeros_like(Ny), where=norm != 0)
    Nz = np.divide(Nz, norm, out=np.zeros_like(Nz), where=norm != 0)

    # Scale to 0-255 range and convert to uint8
    def scale(vec):
        scaled = np.clip((vec + 1) / 2, 0, 1) * 255
        return scaled.astype(np.uint8)

    scaled_Nx = scale(Nx)
    scaled_Ny = scale(Ny)
    scaled_Nz = scale(Nz)

    # Create 3-channel normal image
    normal_image = cv2.merge([scaled_Nx, scaled_Ny, scaled_Nz])

    return normal_image


def process_single_image(depth_data: ImageData) -> ImageData:
    """
    Process a single depth image by computing normals.

    Args:
        depth_data (ImageData): An ImageData object containing the depth image and timestamp.

    Returns:
        ImageData: An ImageData object containing the normal image and the same timestamp.
    """
    # Compute normals from the depth image
    normal_image = compute_normals(depth_data.image)
    return ImageData(image=normal_image, timestamp=depth_data.timestamp)


def convert_depth_images_to_normal_images(
    depth_images_list: List[ImageData], num_processes: int = None
) -> List[ImageData]:
    """
    Convert a list of depth images to normal images using multiprocessing.

    Args:
        depth_images_list (List[ImageData]): List of depth image data.
        num_processes (int, optional): Number of processes to use. If None, it will use the number of CPU cores.

    Returns:
        List[ImageData]: List of normal image data.
    """
    total_images = len(depth_images_list)

    # Number of processes to use
    if num_processes is None:
        num_processes = cpu_count()

    console = Console()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style="yellow"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        convert_task = progress.add_task(
            "[cyan]🩻  Converting depth images to normal images", total=total_images
        )

        # Use multiprocessing to process images in parallel
        with Pool(processes=num_processes) as pool:
            normal_image_data_list = []
            for normal_image_data in pool.imap(process_single_image, depth_images_list):
                normal_image_data_list.append(normal_image_data)
                progress.update(convert_task, advance=1)

    console.print(
        f"[bold green]Successfully converted {len(normal_image_data_list)} out of {total_images} depth images to normal images[/bold green]"
    )

    return normal_image_data_list
