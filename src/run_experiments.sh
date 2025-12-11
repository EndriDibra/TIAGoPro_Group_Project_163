#!/bin/bash
# =============================================================================
# Automated Experiment Runner for TIAGo Social Navigation
# Runs 4 experiments sequentially, each for 1 hour (configurable)
# =============================================================================

set -e  # Exit on error

# Configuration
EXPERIMENT_DURATION=600  # 10 minutes in seconds
LOG_DIR="$HOME/src/tests/logs"
ROSBAG_DIR="$HOME/src/tests/rosbags"
WORKSPACE_DIR="$HOME/TIAGoPro_Group_Project_163"

# Base launch command
BASE_CMD="ros2 launch tiago_social_sim simulation.launch.py navigation:=True slam:=True record:=True world_name:=simple_office"

# Experiments: name -> extra args
declare -A EXPERIMENTS=(
    ["mistral"]="vlm_backend:=mistral"
    ["smol"]="vlm_backend:=smol"
    ["novlm"]="vlm_backend:=mock"
    ["notrack"]="vlm_backend:=mock track_humans:=False"
)

# Experiment order (bash associative arrays don't preserve order)
EXPERIMENT_ORDER=("mistral" "smol" "novlm" "notrack")

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

cleanup() {
    log "Cleaning up ROS2 processes..."
    # Kill all ROS2 related processes gracefully
    pkill -SIGINT -f "ros2" 2>/dev/null || true
    pkill -SIGINT -f "gzserver" 2>/dev/null || true
    pkill -SIGINT -f "gzclient" 2>/dev/null || true
    sleep 5
    # Force kill if still running
    pkill -9 -f "ros2" 2>/dev/null || true
    pkill -9 -f "gzserver" 2>/dev/null || true
    pkill -9 -f "gzclient" 2>/dev/null || true
    sleep 2
    log "Cleanup complete"
}

run_experiment() {
    local name=$1
    local extra_args=$2
    local log_file="$LOG_DIR/${name}.log"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    
    log "=========================================="
    log "Starting experiment: $name"
    log "Duration: $((EXPERIMENT_DURATION / 60)) minutes"
    log "Extra args: $extra_args"
    log "Log file: $log_file"
    log "=========================================="
    
    # Ensure clean state
    cleanup
    
    # Create log directory if needed
    mkdir -p "$LOG_DIR"
    mkdir -p "$ROSBAG_DIR"
    
    # Source ROS2 workspace
    source /opt/ros/humble/setup.bash
    source "$WORKSPACE_DIR/install/setup.bash"
    
    # Run the experiment in background
    $BASE_CMD $extra_args &> "$log_file" &
    local pid=$!
    
    log "Experiment started with PID: $pid"
    log "Waiting for $((EXPERIMENT_DURATION / 60)) minutes..."
    
    # Wait for the experiment duration
    sleep $EXPERIMENT_DURATION
    
    log "Time's up! Stopping experiment: $name"
    
    # Graceful shutdown
    cleanup
    
    # Rename rosbag with experiment name for clarity
    local latest_bag=$(ls -td "$ROSBAG_DIR"/social_nav* 2>/dev/null | head -1)
    if [ -n "$latest_bag" ]; then
        mv "$latest_bag" "$ROSBAG_DIR/${name}_${timestamp}"
        log "Rosbag saved: $ROSBAG_DIR/${name}_${timestamp}"
    fi
    
    log "Experiment $name completed!"
    log ""
    
    # Brief pause between experiments
    sleep 10
}

# =============================================================================
# Main Execution
# =============================================================================

# Trap Ctrl+C to cleanup
trap cleanup EXIT

log "=========================================="
log "TIAGo Social Navigation Experiment Suite"
log "=========================================="
log "Total experiments: ${#EXPERIMENT_ORDER[@]}"
log "Duration per experiment: $((EXPERIMENT_DURATION / 60)) minutes"
log "Total estimated time: $((EXPERIMENT_DURATION * ${#EXPERIMENT_ORDER[@]} / 3600)) hours"
log ""

# Run each experiment
for name in "${EXPERIMENT_ORDER[@]}"; do
    run_experiment "$name" "${EXPERIMENTS[$name]}"
done

log "=========================================="
log "All experiments completed!"
log "=========================================="
log "Logs saved in: $LOG_DIR"
log "Rosbags saved in: $ROSBAG_DIR"
log ""
log "Experiments run:"
for name in "${EXPERIMENT_ORDER[@]}"; do
    log "  - $name"
done
