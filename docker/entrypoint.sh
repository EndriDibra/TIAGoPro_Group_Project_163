#!/bin/bash
# set -e  <-- Disabled for debugging

# Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# Check if workspace is built. If not, build it.
if [ ! -f "/home/user/install/setup.bash" ]; then
    echo "Workspace not built. Building now... (This may take a minute)"
    # Ensure we are in the workspace root
    cd /home/user
    colcon build --symlink-install || echo "Build failed, but continuing for debug..."
fi

# Source the local workspace
if [ -f "/home/user/install/setup.bash" ]; then
    source /home/user/install/setup.bash
fi

# Execute the command
exec "$@"
