#!/bin/bash
# =============================================================================
# Host-Based Experiment Runner for TIAGo Social Navigation
# Runs experiments by restarting the Docker container between each one
# for complete isolation (clean SLAM, TF cache, etc.)
# =============================================================================

set -e  # Exit on error

# Configuration
EXPERIMENT_DURATION=3600 # seconds
LOG_DIR="./src/tests/logs"
ROSBAG_DIR="./src/tests/rosbags"
CONTAINER_NAME="tiago_group_sim"
SERVICE_NAME="tiago_sim"

# Base launch command (executed inside container)
BASE_CMD="ros2 launch tiago_social_sim simulation.launch.py navigation:=True slam:=True record:=True world_name:=simple_office"

# Experiments: name -> extra args
declare -A EXPERIMENTS=(
    ["novlm"]="vlm_backend:=mock"
    ["smol"]="vlm_backend:=smol"
    ["mistral"]="vlm_backend:=mistral"
    ["notrack"]="vlm_backend:=mock track_humans:=False"
)

# Experiment order
EXPERIMENT_ORDER=("novlm" "smol" "mistral" "notrack")

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

stop_all() {
    log "Stopping all containers..."
    docker compose down 2>/dev/null || true
    docker rm -f $CONTAINER_NAME 2>/dev/null || true
    sleep 2
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
    stop_all
    
    # Create directories
    mkdir -p "$LOG_DIR"
    mkdir -p "$ROSBAG_DIR"
    
    # Start container with simulation command
    # Note: bash -c starts a new shell that doesn't inherit entrypoint's sourced env
    # We must source: 1) ROS humble, 2) PAL system, 3) our workspace
    log "Starting container with simulation..."
    docker compose run --rm --name $CONTAINER_NAME \
        -e DISPLAY=$DISPLAY \
        $SERVICE_NAME \
        bash -c "source /opt/ros/humble/setup.bash && source /opt/pal/alum/setup.bash && source /home/user/install/setup.bash && $BASE_CMD $extra_args" &> "$log_file" &
    
    local docker_pid=$!
    
    log "Experiment started (docker PID: $docker_pid)"
    log "Waiting for $((EXPERIMENT_DURATION / 60)) minutes..."
    
    # Wait for duration
    sleep $EXPERIMENT_DURATION
    
    log "Time's up! Stopping experiment: $name"
    
    # Stop container
    stop_all
    
    # Rename rosbag
    local latest_bag=$(ls -td "$ROSBAG_DIR"/[0-9]*_[0-9]* 2>/dev/null | head -1)
    if [ -n "$latest_bag" ]; then
        mv "$latest_bag" "$ROSBAG_DIR/${name}_${timestamp}"
        log "Rosbag saved: $ROSBAG_DIR/${name}_${timestamp}"
    fi
    
    log "Experiment $name completed!"
    log ""
    
    sleep 5
}

# =============================================================================
# Main Execution
# =============================================================================

trap stop_all EXIT

LOOP_COUNT=3  # Number of times to repeat all experiments

log "=========================================="
log "TIAGo Social Navigation Experiment Suite"
log "(Host-Based Container Restart Mode)"
log "=========================================="
log "Total experiments per loop: ${#EXPERIMENT_ORDER[@]}"
log "Number of loops: $LOOP_COUNT"
log "Total experiments: $((${#EXPERIMENT_ORDER[@]} * LOOP_COUNT))"
log "Duration per experiment: $((EXPERIMENT_DURATION / 60)) minutes"
log "Total estimated runtime: $(( ${#EXPERIMENT_ORDER[@]} * LOOP_COUNT * EXPERIMENT_DURATION / 3600 )) hours"
log ""

for ((loop=1; loop<=LOOP_COUNT; loop++)); do
    log "=========================================="
    log "LOOP $loop of $LOOP_COUNT"
    log "=========================================="
    
    for name in "${EXPERIMENT_ORDER[@]}"; do
        run_experiment "$name" "${EXPERIMENTS[$name]}"
    done
done

log "=========================================="
log "All experiments completed!"
log "=========================================="
log "Loops completed: $LOOP_COUNT"
log "Total experiments: $((${#EXPERIMENT_ORDER[@]} * LOOP_COUNT))"
log "Logs saved in: $LOG_DIR"
log "Rosbags saved in: $ROSBAG_DIR"
