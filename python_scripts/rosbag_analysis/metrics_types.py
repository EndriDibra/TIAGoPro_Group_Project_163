"""Data classes for rosbag metrics extraction."""
from dataclasses import dataclass, field
from typing import Optional
import csv


@dataclass
class ScenarioMetrics:
    """Metrics computed for a single scenario run."""
    experiment_name: str      # e.g., "mistral", "smol", "novlm", "notrack"
    scenario_name: str        # e.g., "frontal_approach", "intersection", "narrow_doorway"
    scenario_run: int         # Run number (1, 2, 3...) for repeated scenarios
    
    # Core metrics
    psc: float                # Personal Space Compliance (% timesteps >= 1.25m edge-to-edge)
    min_ttc: float            # Minimum Time-to-Collision (seconds), inf if never approaching
    min_distance: float       # Minimum edge-to-edge distance to human (meters)
    
    # Latency metrics
    detection_latency: float  # First /human/odom -> first person detection (seconds)
    vlm_latency: float        # First detection -> VLM response (seconds)
    
    # VLM action counts
    continue_count: int = 0   # Number of "Continue" actions from VLM
    slow_down_count: int = 0  # Number of "Slow Down" actions from VLM
    yield_count: int = 0      # Number of "Yield" actions from VLM
    
    # Duration
    scenario_duration: float = 0.0  # Total scenario time (seconds)
    
    # Metadata
    start_time: float = 0.0   # Scenario start timestamp (for reference)
    end_time: float = 0.0     # Scenario end timestamp (for reference)


@dataclass
class TimestampedPose:
    """A pose with timestamp for synchronization."""
    timestamp: float  # In seconds
    x: float
    y: float
    yaw: float = 0.0  # Orientation in radians (needed for robot footprint)
    vx: float = 0.0   # Velocity x
    vy: float = 0.0   # Velocity y


def write_metrics_csv(metrics: list[ScenarioMetrics], output_path: str):
    """Write metrics to CSV file."""
    fieldnames = [
        'experiment', 'scenario', 'run', 
        'psc', 'min_ttc', 'min_distance',
        'detection_latency', 'vlm_latency',
        'continue_count', 'slow_down_count', 'yield_count',
        'scenario_duration'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for m in metrics:
            writer.writerow({
                'experiment': m.experiment_name,
                'scenario': m.scenario_name,
                'run': m.scenario_run,
                'psc': f'{m.psc:.2f}',
                'min_ttc': f'{m.min_ttc:.3f}' if m.min_ttc != float('inf') else 'inf',
                'min_distance': f'{m.min_distance:.3f}' if m.min_distance != float('inf') else 'inf',
                'detection_latency': f'{m.detection_latency:.3f}' if m.detection_latency != -1.0 else 'N/A',
                'vlm_latency': f'{m.vlm_latency:.3f}' if m.vlm_latency != -1.0 else 'N/A',
                'continue_count': m.continue_count,
                'slow_down_count': m.slow_down_count,
                'yield_count': m.yield_count,
                'scenario_duration': f'{m.scenario_duration:.2f}'
            })
    
    print(f"Wrote {len(metrics)} metrics to {output_path}")
