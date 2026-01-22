# Presentation Analysis Scripts

This folder contains comprehensive presentation analysis scripts for the TIAGoPro social navigation project.

## Presentation Color Scheme

- Cyan: `#01ffff`
- Blue: `#0a00e9`
- Green: `#3fff45`
- Pink: `#fc38db`
- Black and white as needed

## Scripts Overview

### Task 1: PSC Violin Plot
**Script:** [`task1_psc_violin.py`](task1_psc_violin.py)
**Output:** `psc_violin.svg`

Creates a violin plot showing PSC (Personal Space Compliance) distribution for each experiment.

**Usage:**
```bash
python3 task1_psc_violin.py
```

### Task 2: Collision Heatmap
**Script:** [`task2_collision_heatmap.py`](task2_collision_heatmap.py)
**Output:** `collision_heatmap.svg`

Creates a collision heatmap showing collision rate by experiment and scenario using presentation colors.

**Usage:**
```bash
python3 task2_collision_heatmap.py
```

### Task 3: Velocity Estimation Accuracy (Scatter Plot)
**Script:** [`task3_velocity_accuracy.py`](task3_velocity_accuracy.py)
**Output:** `velocity_accuracy.svg`

Creates a scatter plot comparing predicted vs. actual human velocities. 
Extracts human velocity from `/human/odom` (actual) and tracked velocity from `/social_costmap/person_markers` (predicted).

**Requires ROS2 environment.**

**Usage (in ROS2 environment):**
```bash
# Option 1: Using docker container
docker exec -it tiago_group_sim bash
cd /home/user/python_scripts/presentation
python3 task3_velocity_accuracy.py

# Option 2: Source ROS2 and run
source /opt/ros/humble/setup.bash
python3 task3_velocity_accuracy.py
```

### Task 4: Tracking Performance (Violin Plot)
**Script:** [`task4_tracking_performance.py`](task4_tracking_performance.py)
**Output:** `tracking_performance.svg`

Creates a violin plot of tracking error per scenario per experiment.
Extracts real human positions from `/human/odom` and tracked positions from `/social_costmap/person_markers`.

**Requires ROS2 environment.**

**Usage (in ROS2 environment):**
```bash
python3 task4_tracking_performance.py
```

### Task 5: VLM Metrics Bar Chart
**Script:** [`task5_vlm_metrics_bar.py`](task5_vlm_metrics_bar.py)
**Output:** `vlm_metrics_bar.svg`

Creates a bar chart showing Grounding (human and LLM), Consistency (human and LLM), and Appropriateness.
Data is parsed from `vlm_metrics_summary.txt`.

**Usage:**
```bash
python3 task5_vlm_metrics_bar.py
```

### Task 6: VLM Actions Stacked Bar
**Script:** [`task6_vlm_actions_stacked.py`](task6_vlm_actions_stacked.py)
**Output:** `vlm_actions_stacked.svg`

Creates a stacked bar chart showing VLM action distribution (Continue, Slow Down, Yield) by experiment.

**Usage:**
```bash
python3 task6_vlm_actions_stacked.py
```

### Task 7: Energy at Collision (Violin Plot)
**Script:** [`task7_energy_collision.py`](task7_energy_collision.py)
**Output:** `energy_collision.svg`

Creates a violin plot showing energy at collision per experiment.
Identifies collision events (min_distance < 0), extracts velocities at collision from `/human/odom` and `/mobile_base_controller/odom`, and calculates kinetic energy.

**Requires ROS2 environment.**

**Injury thresholds:**
- Minor injury: ~30 Joules
- Moderate injury: ~100 Joules
- Serious injury: >200 Joules

**Usage (in ROS2 environment):**
```bash
python3 task7_energy_collision.py
```

### Task 8: Mann-Whitney U Test
**Script:** [`task8_mann_whitney_test.py`](task8_mann_whitney_test.py)
**Output:** `mann_whitney_results.txt`

Computes Mann-Whitney U test comparing No VLM vs. No Tracking for:
- PSC (Personal Space Compliance)
- Min Distance
- Collision rate

Provides full precision (no rounding) in the output.

**Usage:**
```bash
python3 task8_mann_whitney_test.py
```

## Running All Non-ROS2 Scripts

To run all scripts that don't require ROS2:

```bash
cd python_scripts/presentation
python3 task1_psc_violin.py
python3 task2_collision_heatmap.py
python3 task5_vlm_metrics_bar.py
python3 task6_vlm_actions_stacked.py
python3 task8_mann_whitney_test.py
```

## Running ROS2-Dependent Scripts

The following scripts require a ROS2 environment with rosbag2_py:

- [`task3_velocity_accuracy.py`](task3_velocity_accuracy.py)
- [`task4_tracking_performance.py`](task4_tracking_performance.py)
- [`task7_energy_collision.py`](task7_energy_collision.py)

### Using Docker Container

```bash
# Start the container (if not already running)
docker-compose up -d

# Enter the container
docker exec -it tiago_group_sim bash

# Navigate to presentation folder
cd /home/user/python_scripts/presentation

# Run the scripts
python3 task3_velocity_accuracy.py
python3 task4_tracking_performance.py
python3 task7_energy_collision.py
```

### Using Local ROS2 Installation

```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Navigate to presentation folder
cd python_scripts/presentation

# Run the scripts
python3 task3_velocity_accuracy.py
python3 task4_tracking_performance.py
python3 task7_energy_collision.py
```

## Output Files

All plots are saved as SVG files in this directory (`python_scripts/presentation/`):

- `psc_violin.svg` - PSC distribution by experiment
- `collision_heatmap.svg` - Collision rate heatmap
- `velocity_accuracy.svg` - Velocity estimation accuracy (requires ROS2)
- `tracking_performance.svg` - Tracking error by scenario (requires ROS2)
- `vlm_metrics_bar.svg` - VLM evaluation metrics
- `vlm_actions_stacked.svg` - VLM action distribution
- `energy_collision.svg` - Energy at collision (requires ROS2)
- `mann_whitney_results.txt` - Statistical test results

## Data Sources

- **Metrics CSV:** `../rosbag_analysis/final_data/final_metrics.csv`
- **VLM Metrics:** `../rosbag_analysis/final_data/vlm_metrics_summary.txt`
- **Rosbag Files:** `../../src/tests/rosbags/`

## Dependencies

### Python Packages
- pandas
- numpy
- matplotlib
- scipy
- seaborn

### ROS2 Packages (for tasks 3, 4, 7)
- rosbag2_py
- rclpy
- rosidl_runtime_py

## Constants Used

- `PSC_THRESHOLD = 1.25` meters
- `HUMAN_RADIUS = 0.25` meters
- `ROBOT_HALF_LENGTH = 0.35` meters
- `ROBOT_HALF_WIDTH = 0.24` meters
- `ROBOT_MASS = 80` kg
- `HUMAN_MASS = 70` kg
- `SYNC_TOLERANCE = 0.2` seconds

## Experiment Names

| Key | Display Name | Color |
|-----|--------------|--------|
| mistral | Cloud VLM | Blue (#0a00e9) |
| smol | Local VLM | Cyan (#01ffff) |
| novlm | No VLM | Green (#3fff45) |
| notrack | No Tracking | Pink (#fc38db) |

## Scenario Names

| Key | Display Name |
|-----|--------------|
| frontal_approach | Frontal Approach |
| intersection | Intersection |
| doorway | Doorway |
