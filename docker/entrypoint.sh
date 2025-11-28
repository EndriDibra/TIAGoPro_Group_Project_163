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

# Deploy PAL user configuration overrides
echo "Deploying PAL user configuration overrides..."
mkdir -p /home/user/.pal/config
if [ -d "/home/user/src/tiago_social_sim/pal_config" ]; then
    cp -v /home/user/src/tiago_social_sim/pal_config/*.yaml /home/user/.pal/config/
    echo "PAL config overrides deployed successfully"
else
    echo "Warning: PAL config source directory not found"
fi

# Source the local workspace
if [ -f "/home/user/install/setup.bash" ]; then
    source /home/user/install/setup.bash
fi

# Execute the command
exec "$@"
