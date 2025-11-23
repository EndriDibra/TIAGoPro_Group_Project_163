# TIAGo Pro Navigation Stack Summary

This document provides a comprehensive overview of the navigation stack running on the TIAGo Pro robot (simulated). It details the architecture, custom modifications, and the standard configuration provided by PAL Robotics.

## 1. High-Level Architecture

The navigation stack is built on **ROS 2 Humble** and **Nav2**. It follows a modular launch structure:

1.  **Entry Point**: `tiago_social_sim/launch/simulation.launch.py`
    *   This is the custom launch file used to start the simulation.
    *   It handles robot spawning, Gazebo world, and brings up the navigation stack.
2.  **Navigation Launch**: `omni_base_2dnav/launch/navigation.launch.py`
    *   Included by the entry point.
    *   Loads the standard Nav2 nodes: `planner_server`, `controller_server`, `bt_navigator`, `recoveries_server`, `lifecycle_manager`.
    *   It uses **PAL Navigation Configuration** (`pal_navigation_cfg`) to load presets based on the robot type (`omni_base`).

## 2. Custom Modifications

The `tiago_social_sim` package includes several important modifications to the standard stack, primarily defined in `simulation.launch.py`:

### 2.1. EKF Odometry Fusion
A **Robot Localization (EKF)** node is added to fuse odometry data. This is critical for accurate localization, especially when wheel slip occurs.
*   **Package**: `robot_localization`
*   **Node**: `ekf_node`
*   **Configuration**:
    *   **Mode**: 2D (x, y, yaw)
    *   **Input**: `/mobile_base_controller/odom` (Wheel Odometry only).
    *   **Note**: No IMU or visual odometry is currently fused in this configuration.
    *   **Frames**: `map` -> `odom` -> `base_footprint`
    *   **Frequency**: 30 Hz

### 2.2. WSL2 Compatibility Fixes
Several fixes are implemented to ensure stability in WSL2 environments:
*   **Cleanup**: Removes lock files and kills stale Gazebo processes before startup.
*   **Topic Remapping**: Relays `/scan_front_raw` to `/scan` for compatibility.
*   **Delays**: Adds a 15-second delay before bringing up the robot to allow Gazebo to stabilize.

## 3. Standard Configuration (PAL Defaults)

The core navigation logic uses the `omni_base` configuration from PAL Robotics.

### 3.1. Controller (Local Planner)
The robot uses the **MPPI (Model Predictive Path Integral)** controller. This is a high-performance, predictive controller suitable for dynamic environments and omnidirectional robots.

*   **Plugin**: `nav2_mppi_controller::MPPIController`
*   **Key Settings**:
    *   **Time Steps**: 56 (Lookahead horizon)
    *   **Model DT**: 0.05s
    *   **Batch Size**: 2000 (Number of trajectories sampled per step)
    *   **Critics** (Cost Functions):
        *   `ConstraintCritic`: Avoids kinematic limits.
        *   `GoalCritic` & `GoalAngleCritic`: Guides towards the goal.
        *   `PathAlignCritic` & `PathFollowCritic`: Keeps the robot on the global path.
        *   `PreferForwardCritic`: Encourages forward motion.
        *   `ObstacleCritic`: Avoids obstacles (via costmap).

### 3.2. Global Planner
The global planner computes the path from start to goal.
*   **Plugin**: **SmacPlanner2D** (`nav2_smac_planner/SmacPlanner2D`).
*   **Configuration**: Defined in `00_planner_server.yaml` via the `smac_2D.yaml` preset.
*   **Algorithm**: A* search on a 2D grid (costmap). It supports smooth paths and is well-suited for omnidirectional robots.

### 3.3. Costmaps
Costmaps are 2D grids representing safe and unsafe areas.

#### Local Costmap (for Controller)
*   **Type**: Rolling Window (moves with the robot).
*   **Size**: 5m x 5m.
*   **Resolution**: 0.05m (5cm).
*   **Update Rate**: 5 Hz.
*   **Plugins**:
    *   `obstacle_laser_layer`: Updates from LaserScan (`/scan`).
    *   `inflation_layer`: Adds a safety buffer around obstacles.

#### Global Costmap (for Planner)
*   **Type**: Static (fixed to the map).
*   **Plugins**:
    *   `static_layer`: Loads the map from the map server.
    *   `obstacle_laser_layer`: Updates dynamic obstacles from LaserScan.
    *   `inflation_layer`: Adds safety buffer.
*   **Raytracing**: Clears obstacles up to 15m.

### 3.4. Localization & SLAM
*   **Modes**: The robot runs in either **Localization (AMCL)** OR **SLAM** mode, but never both simultaneously.
    *   **AMCL**: Used when `slam:=False` (default). Estimates position on a static map.
    *   **SLAM**: Used when `slam:=True`. Builds a map and localizes simultaneously.
*   **SLAM Package**: **Slam Toolbox** (`slam_toolbox`).
    *   **Node**: `sync_slam_toolbox_node`.
    *   **Function**: Performs online SLAM and map saving.

## 4. "Behind the Scenes" Workflow

When you send a goal (e.g., via RViz 2D Nav Goal):

1.  **BT Navigator**: Receives the goal and executes the Behavior Tree (XML).
2.  **Planner Server**:
    *   Uses **Global Costmap** to calculate a path from current pose to goal.
    *   Passes the path to the Controller.
3.  **Controller Server (MPPI)**:
    *   Receives the path.
    *   Uses **Local Costmap** to perceive immediate surroundings.
    *   Simulates thousands of random trajectories (Batch Size 2000).
    *   Scores them based on "Critics" (distance to path, obstacle proximity, goal alignment).
    *   Selects the best velocity command (`cmd_vel`).
4.  **Mobile Base**: Executes the `cmd_vel` to move the wheels.
5.  **EKF & AMCL**: Continuously update the robot's estimated position to keep the loop closed.
