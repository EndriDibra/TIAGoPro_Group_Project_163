#!/bin/bash
# =============================================================================
# Automated Experiment Runner for TIAGo Social Navigation
# Runs 4 experiments sequentially, each for 1 hour (configurable)
# =============================================================================

set -e  # Exit on error

# Configuration
EXPERIMENT_DURATION=600  # 10 minutes in seconds
LOG_DIR="./src/tests/logs"
ROSBAG_DIR="./src/tests/rosbags"

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

# Global variable to track experiment process group
EXPERIMENT_PGID=""

cleanup_experiment() {
    log "Cleaning up experiment..."
    # Kill only the experiment's process group if it exists
    if [ -n "$EXPERIMENT_PGID" ] && [ "$EXPERIMENT_PGID" != "0" ]; then
        log "Stopping experiment process group: $EXPERIMENT_PGID"
        kill -SIGINT -$EXPERIMENT_PGID 2>/dev/null || true
        sleep 3
        kill -SIGTERM -$EXPERIMENT_PGID 2>/dev/null || true
        sleep 2
        kill -9 -$EXPERIMENT_PGID 2>/dev/null || true
        EXPERIMENT_PGID=""
    fi
    log "Cleanup complete"
}

run_experiment() {
    local name=$1
    local extra_args=$2
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local log_file="$LOG_DIR/${name}_${timestamp}.log"
    
    log "=========================================="
    log "Starting experiment: $name"
    log "Duration: $((EXPERIMENT_DURATION / 60)) minutes"
    log "Extra args: $extra_args"
    log "Log file: $log_file"
    log "=========================================="
    
    # Ensure clean state
    cleanup_experiment
    
    # Create log directory if needed
    mkdir -p "$LOG_DIR"
    mkdir -p "$ROSBAG_DIR"
    
    # Source ROS2 workspace
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    
    # Run the experiment in its own process group for clean termination
    setsid $BASE_CMD $extra_args &> "$log_file" &
    local pid=$!
    EXPERIMENT_PGID=$(ps -o pgid= -p $pid 2>/dev/null | tr -d ' ')
    
    log "Experiment started with PID: $pid, PGID: $EXPERIMENT_PGID"
    log "Waiting for $((EXPERIMENT_DURATION / 60)) minutes..."
    
    # Wait for the experiment duration
    sleep $EXPERIMENT_DURATION
    
    log "Time's up! Stopping experiment: $name"
    
    # Graceful shutdown
    cleanup_experiment
    
    # Rename rosbag with experiment name for clarity
    local latest_bag=$(ls -td "$ROSBAG_DIR"/[0-9]*_[0-9]* 2>/dev/null | head -1)
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
