import asyncio
import concurrent.futures
import os
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from core.data_models import ImageData

from core.models.depth_anything_v2.dpt import DepthAnythingV2
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    """
    A custom Dataset for images that resizes them to have dimensions
    that are multiples of 14, as required by the DepthAnythingV2 model.

    Args:
        image_data_list (List[ImageData]): List of ImageData objects containing images and timestamps.
        target_size (int): Target size to resize images to. If -1, images are resized to closest multiples of 14.
    """

    def __init__(self, image_data_list: List[ImageData], target_size: int = -1):
        self.image_data_list = image_data_list
        self.target_size = target_size

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.image_data_list)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image tensor, original size, and timestamp.
        """
        image_data = self.image_data_list[idx]
        original_image = image_data.image
        original_height, original_width = original_image.shape[:2]

        # Calculate new dimensions that are multiples of 14
        resize_width, resize_height = calculate_resize_dims(
            original_height, original_width, self.target_size
        )

        # Resize the image to dimensions that are multiples of 14
        resized_image = cv2.resize(original_image, (resize_width, resize_height))

        # Convert the image to a torch.Tensor and normalize
        image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image_tensor,
            "original_size": torch.tensor([original_height, original_width]),
            "timestamp": image_data.timestamp,
        }


async def apply_depth_anything_v2(
    image_data_list: List[ImageData],
    encoder: str = "vitl",
    batch_size: int = 4,
    target_size: int = -1,
    num_processes: int = None,
) -> List[ImageData]:
    """
    Apply the DepthAnythingV2 model to a list of images asynchronously.

    Args:
        image_data_list (List[ImageData]): List of ImageData objects containing images and timestamps.
        encoder (str): Type of encoder to use for the model. One of 'vits', 'vitb', 'vitl', 'vitg'.
        batch_size (int): Number of images to process in each batch.
        target_size (int): Target size to resize images to. If -1, images are resized to dimensions that are multiples of 14.
        num_processes (int, optional): Number of processes to use. If None, it will use the number of CPU cores.

    Returns:
        List[ImageData]: List of ImageData objects containing the depth images and timestamps.
    """
    # Determine the device to use
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Number of processes to use. If None, it will use the number of CPU cores.
    if num_processes is None:
        num_processes = cpu_count()

    # Model configurations based on the selected encoder
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    # Initialize the DepthAnythingV2 model
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(
        torch.load(
            f"core/models/depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth",
            map_location="cpu",
            weights_only=True,
        )
    )
    depth_anything = depth_anything.to(DEVICE).eval()

    # Create the dataset and dataloader
    dataset = ImageDataset(image_data_list, target_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_processes, pin_memory=True
    )

    depth_image_data_list: List[ImageData] = []

    console = Console()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style="yellow"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Create a progress task
        progress_task = progress.add_task(
            f"[cyan]🧠 Apply Depth Anything V2", total=len(image_data_list)
        )

        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_processes)

        async def process_batch(batch: dict):
            """
            Process a batch of images through the depth model.

            Args:
                batch (dict): Batch data containing images, original sizes, and timestamps.
            """
            images = batch["image"].to(DEVICE)
            original_sizes = batch["original_size"]
            timestamps = batch["timestamp"]

            with torch.no_grad():
                depth_batch = depth_anything(images)

            depth_batch = depth_batch.cpu().numpy()

            # Process resizes concurrently
            tasks = []
            for depth, orig_size in zip(depth_batch, original_sizes):
                orig_width, orig_height = get_image_size(orig_size)
                # Depth maps need to be resized back to original dimensions
                tasks.append(
                    loop.run_in_executor(
                        executor,
                        resize_image,
                        depth.squeeze(),
                        (orig_width, orig_height),
                    )
                )

            depth_resized_list = await asyncio.gather(*tasks)

            for depth_resized, timestamp in zip(depth_resized_list, timestamps):
                depth_image_data_list.append(
                    ImageData(image=depth_resized, timestamp=timestamp)
                )
                progress.update(progress_task, advance=1)

        # Process each batch in the dataloader
        for batch in dataloader:
            await process_batch(batch)

    console.print(
        f"[bold green]Successfully applied Depth Anything V2 on {len(depth_image_data_list)} out of {len(image_data_list)} images[/bold green]"
    )

    return depth_image_data_list


def visualize_depth_map(
    depth_map: np.ndarray, colormap: str = "viridis", normalize: bool = True
) -> np.ndarray:
    """
    Visualize a depth map using a specified colormap.

    Args:
        depth_map (np.ndarray): Raw depth map.
        colormap (str): Name of the matplotlib colormap to use.
        normalize (bool): Whether to normalize the depth map before visualization.

    Returns:
        np.ndarray: Colored depth map as a uint8 BGR image suitable for OpenCV.
    """
    # Replace NaN and infinite values with 0
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        min_val = depth_map.min()
        max_val = depth_map.max()
        # Avoid normalization if min and max are the same
        if max_val - min_val > 1e-8:
            depth_map = (depth_map - min_val) / (max_val - min_val)
        else:
            depth_map = np.zeros_like(depth_map)
    else:
        depth_map = np.clip(depth_map, 0, 1)

    cmap = plt.get_cmap(colormap)
    colored_map = (cmap(depth_map)[:, :, :3] * 255).astype(np.uint8)

    return colored_map[:, :, ::-1]  # Convert RGB to BGR for OpenCV


def depth_to_grayscale(depth_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Convert a depth map to a grayscale image, handling NaN and infinite values.

    Args:
        depth_map (np.ndarray): Raw depth map.
        normalize (bool): Whether to normalize the depth map before conversion.

    Returns:
        np.ndarray: Grayscale depth map as a uint8 image.
    """
    # Replace NaN and infinite values with 0
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        min_depth = depth_map.min()
        max_depth = depth_map.max()
        # Avoid normalization if min and max are the same
        if max_depth - min_depth > 1e-8:
            depth_map = (depth_map - min_depth) / (max_depth - min_depth)
        else:
            depth_map = np.zeros_like(depth_map)
    else:
        depth_map = np.clip(depth_map, 0, 1)

    # Scale to 0-255 and convert to uint8
    depth_map_uint8 = (depth_map * 255).astype(np.uint8)

    return depth_map_uint8


def grayscale_to_3channel(gray_image: np.ndarray) -> np.ndarray:
    """
    Convert a single-channel grayscale image to a 3-channel grayscale image.

    Args:
        gray_image (np.ndarray): Single-channel grayscale image.

    Returns:
        np.ndarray: 3-channel grayscale image.

    Raises:
        ValueError: If the input is not a 2D array (single-channel image).
    """
    if gray_image.ndim != 2:
        raise ValueError("Input must be a single-channel (2D) grayscale image")

    three_channel = cv2.merge([gray_image, gray_image, gray_image])

    return three_channel


def save_depth_image(
    depth_image: np.ndarray,
    timestamp: float,
    output_folder: str,
    prefix: str = "depth_",
) -> str:
    """
    Save a depth image to a specified folder.

    Args:
        depth_image (np.ndarray): The depth image to save.
        timestamp (float): The timestamp of the image.
        output_folder (str): The folder where the image will be saved.
        prefix (str, optional): Prefix for the filename. Defaults to "depth_".

    Returns:
        str: The path of the saved image file.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Convert timestamp to a formatted string
    timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")

    # Create a unique filename
    filename = f"{prefix}{timestamp_str}.png"
    file_path = os.path.join(output_folder, filename)

    # Ensure the depth image is in the correct format for saving
    if depth_image.dtype != np.uint8:
        # Normalize to 0-255 range only if necessary
        min_val = depth_image.min()
        max_val = depth_image.max()
        if max_val - min_val > 1e-8:
            depth_image_normalized = cv2.normalize(
                depth_image, None, 0, 255, cv2.NORM_MINMAX
            )
            depth_image_uint8 = depth_image_normalized.astype(np.uint8)
        else:
            depth_image_uint8 = np.zeros_like(depth_image, dtype=np.uint8)
    else:
        depth_image_uint8 = depth_image

    # Save the image
    cv2.imwrite(file_path, depth_image_uint8)

    return file_path


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to the specified size using linear interpolation.

    Args:
        image (np.ndarray): Input image.
        size (Tuple[int, int]): Desired size as (width, height).

    Returns:
        np.ndarray: Resized image.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def get_image_size(orig_size: Union[torch.Tensor, List[int], int]) -> Tuple[int, int]:
    """
    Extract the image size from the original_size input.

    Args:
        orig_size (Union[torch.Tensor, List[int], int]): Original size of the image.

    Returns:
        Tuple[int, int]: Original width and height.

    Raises:
        ValueError: If the original_size format is unexpected.
    """
    if isinstance(orig_size, torch.Tensor):
        orig_size = orig_size.tolist()

    if isinstance(orig_size, list):
        if len(orig_size) == 2:
            return int(orig_size[1]), int(orig_size[0])  # width, height
        elif len(orig_size) == 1:
            # Assume it's a square image
            return int(orig_size[0]), int(orig_size[0])
    elif isinstance(orig_size, int):
        return orig_size, orig_size  # Assume square image

    raise ValueError(f"Unexpected original_size format: {orig_size}")


def calculate_resize_dims(
    height: int, width: int, target_size: int = -1
) -> Tuple[int, int]:
    """
    Calculate the dimensions to resize the image to, ensuring they're multiples of 14
    and maintaining the aspect ratio as closely as possible.

    Args:
        height (int): Original height of the image.
        width (int): Original width of the image.
        target_size (int): Target size for the larger dimension. If -1, dimensions are adjusted to be multiples of 14.

    Returns:
        Tuple[int, int]: New width and height, adjusted to be multiples of 14.
    """
    if target_size > 0:
        # Resize based on the target size while maintaining aspect ratio
        aspect_ratio = width / height
        if width > height:
            new_width = target_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(new_height * aspect_ratio)
    else:
        # If target_size is -1, use the original dimensions
        new_height = height
        new_width = width

    # Adjust dimensions to be multiples of 14
    new_width = max(14, (new_width // 14) * 14)
    new_height = max(14, (new_height // 14) * 14)

    # If dimensions are unchanged and already multiples of 14, no resizing needed
    if (
        new_width == width
        and new_height == height
        and new_width % 14 == 0
        and new_height % 14 == 0
    ):
        return width, height
    else:
        return new_width, new_height
