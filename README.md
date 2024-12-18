# ROS-Dataset-Generator

This repository provides a command-line tool to process ROS bag files and optionally generate videos or patches from recorded sensor data. It is designed to be flexible and configurable, allowing you to work with different robot configurations and video generation settings.

## Features

- **Bag File Processing:**  
  Process ROS bag files to extract image data (RGB, depth), odometry, and other sensor readings.
  
- **Video Generation:**  
  Generate various types of videos from your recorded data, such as:
  - RGB video
  - Depth videos (from raw depth data)
  - Videos generated using "Depth Anything" techniques (e.g., normal maps derived from depth)
  - Trajectory overlays on videos
  
- **Patch Extraction (Optional):**  
  Extract patches or specific regions of interest from the dataset for further analysis.
  
- **Robot Configuration Management:**  
  Supports loading of robot configurations (topics, sensor data sources, etc.) from JSON files, making the pipeline adaptable to multiple robot types.

## Repository Structure

.
├─ main.py                           # Entry point for the command-line tool
├─ core/
│  ├─ pipeline/
│  │  ├─ base.py                     # Pipeline base logic
│  │  ├─ nodes/                      # Pipeline nodes
│  ├─ video/
│  │  ├─ video_config.py             # Video generation configuration model
│  │  ├─ video_generation_manager.py # Logic for generating videos
│  │  ├─ factory.py                  # Video pipeline factory
│  ├─ data_models.py                 # Data models for dataset and command-line arguments
│  ├─ tools/
│  │  ├─ bag.py                      # Functions to read data from bag files (RGB, depth, odometry, etc.)
│  │  ├─ robots.py                   # Helper functions for combining odometry and cmd_vel or motor_rpm
│  ├─ robots/
│     ├─ base_robot.py               # Base class for robot configurations
│     ├─ robot_registry.py           # Robot registry for dynamic loading
│     ├─ models/                     # Directory for robot model definitions
└─ robots_configs/                   # Directory containing robot configuration JSON files

## Requirements

- **Python 3.8+**
- **ROS Noetic** or another ROS distribution with `rosbag` support.
- **Pip packages:**  
  - `asyncio`
  - `argparse`
  - `rich`
  - `pathlib`
  - `pyyaml`
  - Other dependencies as specified in your `requirements.txt` or `pyproject.toml`.

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/dbouchabou/ROS-Dataset-Generator.git
   cd ROS-Dataset-Generator
   ```
2. **Install Dependencies:**
  ```bash
   pip install -r requirements.txt
   ```


## Usage

### General Syntax
```bash
python main.py <config_file> <output_dir> [options]
```
### Required Arguments:

- <config_file>: Path to the dataset configuration JSON file.
- <output_dir>: Directory where output videos and patches will be saved.

### Option:

- --generate-video: Enables the video generation process. Requires --video-config.
- --video-config <path>: Path to the video generation YAML configuration (only valid if --generate-video is used).
- --generate-patches: Enables patch extraction.

### Exemple:

```bash
python main.py datasets_configs/ENSTA_asphalt_2024-07-02-16-02-32.json output --generate-video --video-config video_config.yaml
```

This command will:

- Load the dataset configuration from datasets_configs/ENSTA_asphalt_2024-07-02-16-02-32.json.
- Use the robot configuration specified within the dataset config (my_robot_config.json in robots_configs/).
- Generate videos according to the settings in video_config.yaml.
- Save all output (videos, configurations, logs) to the output directory.

## Outputs
- **Videos:**
Generated videos will be saved within the specified output directory, organized by dataset and video type.

**Configurations:**
Copies of the dataset, robot, and video configuration files will be saved for reference.

**Patches (if enabled):**
Extracted patches or images of interest will be saved in a designated subdirectory.

## Troubleshooting
**Missing Dependencies:**
If you encounter ModuleNo**tFoundError, ensure all Python dependencies are installed.

**Configuration Errors:**
Ensure that the dataset configuration JSON, robot configuration JSON, and video configuration YAML files are valid and refer to correct paths and topics.

**ROS Bag Issues:**
Run rosbag info /path/to/your_bag.bag to verify the bag contains the expected topics. Make sure the specified topics match those in your robot configuration.

If you encounter issues, check the console output for error messages and stack traces. Adjust your configurations or data paths accordingly.

## Contributing
Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request with your changes.

## License
This project is licensed under the [MIT License](LICENSE).