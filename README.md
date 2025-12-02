# TIAGoPro Group Project 163

This is our group project from Aalborg University regarding the TIAGo Pro robot, contributing to its Explainable Social Navigation approach.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (optional, for GPU acceleration)
- X server running (for GUI display on WSL/Linux)

## Quick Start (Docker)

We use Docker to ensure a consistent development environment with all dependencies pre-configured.

### 1. Build the Docker Image

```bash
docker compose build
```

This builds the image with:
- ROS 2 Humble
- TIAGo Pro packages from PAL Robotics
- Navigation and SLAM dependencies
- Required packages: `robot_localization`, `launch`, `ament_cmake`

### 2. Start the Container

```bash
docker compose up -d
```

**Quick Rebuild & Restart:**
```bash
# Convenience script to down, build, up, and enter in one command
./restart.sh
```

The container runs in detached mode with:
- NVIDIA GPU support (if available)
- X11 forwarding for GUI applications (Gazebo, RViz)
- Local `src/` folder mounted to `/home/user/src`

### 3. Enter the Container

```bash
docker compose exec -it tiago_sim bash
```

### 4. Launch the Simulation

Inside the container, run:

```bash
ros2 launch tiago_social_sim simulation.launch.py navigation:=True slam:=True
```

The simulation will start:
- **Gazebo** (server and client) - 3D physics simulation
- **Navigation stack** - Nav2 with path planning and obstacle avoidance
- **SLAM** - Simultaneous Localization and Mapping
- **RViz** - Visualization of robot state and sensor data

#### Worlds

Worlds can be found in `/opt/pal/alum/share/pal_gazebo_worlds/worlds`. The following worlds are available:

- aruco
- door_obstacle
- memmo_simple
- pick_place_cabinet
- small_office_w_aruco_w_docks
- ball
- empty
- narrow_passage
- pick_place_close_demo
- small_textured_office
- cabinet_grasping
- empty_room
- objects_on_table
- pick_place_demo
- socmcs
- calibration_room
- factory
- pal_office
- reemc
- stairs
- chairs
- featured
- pal_office_w_aruco
- reemc_indoor
- stamping_demo
- corridor_110cm
- footstep_obstacle
- pal_office_w_docks
- simple_office
- t_shaped
- corridor_60cm
- hit_table
- pal_textured_object
- simple_office_with_people
- tabletop_cube
- corridor_70cm
- home
- passage_110cm
- simple_ramp
- tabletop_cylinder
- corridor_85cm
- hospital
- passage_60cm
- small_factory
- tutorial_office
- dock_and_aruco
- hospital_flat
- passage_70cm
- small_office
- voronoi_graph
- dock_station
- look_to_point
- pg_demo
- small_office_door_obstacle
- willow_garage
- docker_chair
- low_sun_empty
- pick
- small_office_inventory
- world_wall_REEM_photo

### 5. Stop the Container

```bash
docker compose down
```

## Development Workflow

### Building Your Package

Inside the container:

```bash
# Build the workspace
colcon build --symlink-install

# Source the workspace
source install/setup.bash

# Or use the convenient alias (builds and sources)
build
```

### Rebuild After Changes

```bash
# If you modified launch files (using --symlink-install, no rebuild needed)
ros2 launch tiago_social_sim simulation.launch.py navigation:=True slam:=True
ros2 launch tiago_social_sim simulation.launch.py navigation:=True slam:=True world_name:=tutorial_office &> ~/src/tmp/output.log

# If you modified C++ code
colcon build --packages-select tiago_social_sim
source install/setup.bash
```

## Useful Commands

### Robot Control

```bash
# Publish a velocity command to move in circles
ros2 topic pub --once -w 1 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}, angular: {z: 0.5}}"

# Send a navigation goal (requires navigation stack running)
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 0.0, z: 0.0}}}"
```

### ROS 2 Inspection

```bash
# List all active nodes
ros2 node list

# List all topics
ros2 topic list

# View robot sensor data (laser scan)
ros2 topic echo /scan

# Check TF tree
ros2 run tf2_tools view_frames
```

## Docker Configuration

The setup includes:

- **Base Image**: `development-tiago-pro-34:alum-25.01` (PAL Robotics)
- **Runtime**: NVIDIA runtime for GPU acceleration
- **Display**: X11 forwarding configured for WSL/Linux
- **Network**: Host network mode for ROS 2 communication
- **Volumes**: 
  - `./src` → `/home/user/src` (your code)
  - X11 socket for GUI forwarding

## Troubleshooting

### Simulation Crashes or Gazebo Doesn't Start

If you see errors like `gzserver` or `gzclient` dying:

1. **Check X11 forwarding** (WSL/Linux):
   ```bash
   echo $DISPLAY  # Should show :0 or similar
   xauth list     # Should show authentication entries
   ```

2. **Try software rendering** (if GPU issues):
   Edit `docker-compose.yml` and set:
   ```yaml
   environment:
     - LIBGL_ALWAYS_SOFTWARE=1
     - LIBGL_ALWAYS_INDIRECT=1
   ```

3. **Check NVIDIA runtime**:
   ```bash
   docker info | grep -i runtime  # Should show 'nvidia'
   nvidia-smi  # Should work inside container if GPU is available
   ```

### Missing Packages

If you get package not found errors:

```bash
# Inside container, check if package exists
ros2 pkg list | grep <package_name>

# If missing, install it (may need to rebuild Docker image)
sudo apt-get update
sudo apt-get install ros-humble-<package-name>
```

### Shutdown Errors

Segmentation faults during shutdown (e.g., in `move_group` or `component_container`) are known issues with complex ROS 2 stacks and can be safely ignored if the simulation runs correctly.

## Project Structure

```
.
├── docker/
│   ├── Dockerfile          # Container definition
│   └── entrypoint.sh       # Startup script
├── docker-compose.yml      # Docker Compose configuration
├── src/
│   └── tiago_social_sim/   # Your ROS 2 package
│       ├── launch/         # Launch files
│       └── CMakeLists.txt
└── test_sim.sh            # Quick test script
```

## Notes

- The simulation includes WSL2-specific fixes (cleanup scripts, topic remapping, EKF odometry)
- First launch may take 15-20 seconds due to initialization delays
- GPU acceleration improves Gazebo performance but is not required
- Shutdown warnings/errors are normal and can be ignored if simulation runs properly