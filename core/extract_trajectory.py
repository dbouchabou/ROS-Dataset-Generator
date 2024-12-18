import rosbag
import numpy as np
import csv
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
import tf.transformations
from typing import Dict, List, Tuple, Optional
import rospy


class WheelTrajectoryExtractor:
    """Extracts wheel trajectory data in world frame from ROS bag."""

    def __init__(self, bag_path: str):
        self.bag_path = bag_path
        self.console = Console()
        self.tf_data: Dict[Tuple[str, str], List[Tuple[float, np.ndarray]]] = {}
        self.wheel_frame = None
        self.world_frame = None

    def print_bag_info(self) -> None:
        """Print detailed information about bag contents."""
        with rosbag.Bag(self.bag_path) as bag:
            info = bag.get_type_and_topic_info()

            # Calculate duration from message timestamps
            start_time = None
            end_time = None
            for _, _, t in bag.read_messages():
                if start_time is None or t.to_sec() < start_time:
                    start_time = t.to_sec()
                if end_time is None or t.to_sec() > end_time:
                    end_time = t.to_sec()

            duration = end_time - start_time if start_time is not None else 0

            # Create topic table
            table = Table(title="Bag Contents")
            table.add_column("Topic", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Messages", justify="right", style="green")
            table.add_column("Frequency", justify="right", style="yellow")

            for topic_name, topic_info in info.topics.items():
                frequency = topic_info.message_count / duration if duration > 0 else 0
                table.add_row(
                    topic_name,
                    topic_info.msg_type,
                    str(topic_info.message_count),
                    f"{frequency:.2f} Hz",
                )

            self.console.print(table)
            self.console.print(f"\nDuration: {duration:.2f} seconds")

    def find_transform_chain(
        self, source_frame: str, target_frame: str
    ) -> List[Tuple[str, str, bool]]:
        """Find chain of transforms between source and target frames."""
        # Build transform graph
        graph = {}
        for frame1, frame2 in self.tf_data.keys():
            if frame1 not in graph:
                graph[frame1] = set()
            if frame2 not in graph:
                graph[frame2] = set()
            graph[frame1].add(frame2)
            graph[frame2].add(frame1)

        # Use BFS to find path
        queue = [(source_frame, [])]
        visited = {source_frame}

        while queue:
            current_frame, path = queue.pop(0)

            if current_frame == target_frame:
                return path

            for next_frame in graph[current_frame]:
                if next_frame not in visited:
                    visited.add(next_frame)

                    # Check if transform needs to be inverted
                    direct = (current_frame, next_frame) in self.tf_data
                    new_path = path + [(current_frame, next_frame, not direct)]
                    queue.append((next_frame, new_path))

        return []

    def identify_frames(self) -> None:
        """Identify important frames in the transform tree."""
        frames = set()
        for frame1, frame2 in self.tf_data.keys():
            frames.add(frame1)
            frames.add(frame2)

        # Print available frames
        table = Table(title="Available Frames")
        table.add_column("Frame", style="cyan")
        table.add_column("Parent Frames", style="green")
        table.add_column("Child Frames", style="yellow")

        for frame in sorted(frames):
            parents = [f1 for f1, f2 in self.tf_data.keys() if f2 == frame]
            children = [f2 for f1, f2 in self.tf_data.keys() if f1 == frame]
            table.add_row(
                frame, ", ".join(parents) or "None", ", ".join(children) or "None"
            )

        self.console.print(table)

        # Find wheel frame
        for frame in frames:
            if "c1_wheel" in frame and not frame.endswith("support"):
                self.wheel_frame = frame
                break

        # Set world frame
        self.world_frame = "map"

        self.console.print("\n[cyan]Selected Frames:[/cyan]")
        self.console.print(f"World frame: {self.world_frame}")
        self.console.print(f"Wheel frame: {self.wheel_frame}")

        if self.wheel_frame and self.world_frame:
            chain = self.find_transform_chain(self.world_frame, self.wheel_frame)
            if chain:
                self.console.print(
                    "\n[cyan]Transform chain from world to wheel:[/cyan]"
                )
                for source, target, inverse in chain:
                    direction = "<-" if inverse else "->"
                    self.console.print(f"  {source} {direction} {target}")

    def get_transform_at_time(
        self,
        source_frame: str,
        target_frame: str,
        timestamp: float,
        max_time_diff: float = 0.1,
    ) -> Optional[np.ndarray]:
        """Get transform between frames at specific time."""
        key = (source_frame, target_frame)
        if key not in self.tf_data:
            return None

        transforms = self.tf_data[key]
        closest_idx = min(
            range(len(transforms)), key=lambda i: abs(transforms[i][0] - timestamp)
        )

        if abs(transforms[closest_idx][0] - timestamp) > max_time_diff:
            return None

        return transforms[closest_idx][1]

    def get_composite_transform(
        self, source_frame: str, target_frame: str, timestamp: float
    ) -> Optional[np.ndarray]:
        """Get composite transform through transform chain."""
        chain = self.find_transform_chain(source_frame, target_frame)
        if not chain:
            return None

        result = np.eye(4)
        for frame1, frame2, inverse in chain:
            transform = self.get_transform_at_time(frame1, frame2, timestamp)
            if transform is None:
                return None

            if inverse:
                transform = np.linalg.inv(transform)
            result = result @ transform

        return result

    def load_transforms(self) -> None:
        """Load transforms from bag file."""
        try:
            self.console.print("\n[bold cyan]Loading bag file...[/bold cyan]")
            self.print_bag_info()

            with rosbag.Bag(self.bag_path) as bag:
                tf_topics = ["tf", "/tf", "tf_static", "/tf_static"]
                total_msgs = sum(1 for _ in bag.read_messages(topics=tf_topics))

                with Progress() as progress:
                    task = progress.add_task(
                        "[cyan]Loading transforms...", total=total_msgs
                    )

                    for _, msg, t in bag.read_messages(topics=tf_topics):
                        timestamp = t.to_sec()

                        for transform in msg.transforms:
                            key = (transform.header.frame_id, transform.child_frame_id)
                            if key not in self.tf_data:
                                self.tf_data[key] = []

                            # Convert to matrix
                            trans = transform.transform.translation
                            rot = transform.transform.rotation

                            matrix = tf.transformations.quaternion_matrix(
                                [rot.x, rot.y, rot.z, rot.w]
                            )
                            matrix[:3, 3] = [trans.x, trans.y, trans.z]

                            self.tf_data[key].append((timestamp, matrix))

                        progress.update(task, advance=1)

                # Sort transforms by timestamp
                for transforms in self.tf_data.values():
                    transforms.sort(key=lambda x: x[0])

            self.identify_frames()

        except Exception as e:
            self.console.print(f"[bold red]Error loading transforms: {e}[/bold red]")
            raise

    def extract_trajectory(self) -> List[Dict]:
        """Extract wheel trajectory in world frame."""
        if not self.wheel_frame or not self.world_frame:
            self.console.print("[red]Required frames not found[/red]")
            return []

        trajectory_data = []
        timestamps = sorted(
            set(t for transforms in self.tf_data.values() for t, _ in transforms)
        )

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Extracting trajectory...", total=len(timestamps)
            )

            for timestamp in timestamps:
                transform = self.get_composite_transform(
                    self.world_frame, self.wheel_frame, timestamp
                )

                if transform is not None:
                    position = transform[:3, 3]
                    quaternion = tf.transformations.quaternion_from_matrix(transform)
                    euler = tf.transformations.euler_from_matrix(transform)

                    trajectory_data.append(
                        {
                            "timestamp": timestamp,
                            "x": position[0],
                            "y": position[1],
                            "z": position[2],
                            "qx": quaternion[0],
                            "qy": quaternion[1],
                            "qz": quaternion[2],
                            "qw": quaternion[3],
                            "roll": euler[0],
                            "pitch": euler[1],
                            "yaw": euler[2],
                        }
                    )

                progress.update(task, advance=1)

        return trajectory_data

    def save_trajectory(self, output_file: str, trajectory_data: List[Dict]) -> None:
        """Save trajectory data to CSV file."""
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "timestamp",
                    "x_world",
                    "y_world",
                    "z_world",
                    "qx",
                    "qy",
                    "qz",
                    "qw",
                    "roll",
                    "pitch",
                    "yaw",
                ]
            )

            for point in trajectory_data:
                writer.writerow(
                    [
                        f"{point['timestamp']:.6f}",
                        f"{point['x']:.6f}",
                        f"{point['y']:.6f}",
                        f"{point['z']:.6f}",
                        f"{point['qx']:.6f}",
                        f"{point['qy']:.6f}",
                        f"{point['qz']:.6f}",
                        f"{point['qw']:.6f}",
                        f"{point['roll']:.6f}",
                        f"{point['pitch']:.6f}",
                        f"{point['yaw']:.6f}",
                    ]
                )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract wheel trajectory in world frame"
    )
    parser.add_argument("bag_path", help="Path to ROS bag file")
    parser.add_argument("output_path", help="Path for output CSV file")

    args = parser.parse_args()

    console = Console()

    try:
        extractor = WheelTrajectoryExtractor(args.bag_path)
        extractor.load_transforms()

        trajectory_data = extractor.extract_trajectory()

        if trajectory_data:
            extractor.save_trajectory(args.output_path, trajectory_data)
            console.print(
                f"[green]Saved {len(trajectory_data)} points to {args.output_path}[/green]"
            )

            # Print statistics
            timestamps = [p["timestamp"] for p in trajectory_data]
            duration = max(timestamps) - min(timestamps)
            rate = len(trajectory_data) / duration if duration > 0 else 0

            console.print("\n[cyan]Trajectory Statistics:[/cyan]")
            console.print(f"Duration: {duration:.2f} seconds")
            console.print(f"Points: {len(trajectory_data)}")
            console.print(f"Average rate: {rate:.2f} Hz")

            # Calculate ranges
            x_vals = [p["x"] for p in trajectory_data]
            y_vals = [p["y"] for p in trajectory_data]
            z_vals = [p["z"] for p in trajectory_data]

            console.print("\n[cyan]Position Ranges (meters):[/cyan]")
            console.print(f"X: {min(x_vals):.2f} to {max(x_vals):.2f}")
            console.print(f"Y: {min(y_vals):.2f} to {max(y_vals):.2f}")
            console.print(f"Z: {min(z_vals):.2f} to {max(z_vals):.2f}")
        else:
            console.print("[yellow]No trajectory data found[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()
