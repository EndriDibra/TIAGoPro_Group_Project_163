#!/usr/bin/env python3
"""
Task 4: Tracking Performance (Violin Plot)

Creates a violin plot of tracking error per scenario per experiment.
Extracts real human positions from /human/odom and tracked positions from
/social_costmap/person_markers, then computes Euclidean distance error.
"""

import os
import sys
import math
import bisect
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ROS2 imports - must be run in ROS2 environment
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from nav_msgs.msg import Odometry
    from visualization_msgs.msg import MarkerArray
    from std_msgs.msg import String
except ImportError as e:
    print(f"Error: ROS2 packages not found. Run this script in a ROS2 environment.")
    print(f"Details: {e}")
    sys.exit(1)


# Presentation color scheme
PRESENTATION_COLORS = {
    'Cyan': '#01ffff',
    'Blue': '#0a00e9',
    'Green': '#3fff45',
    'Pink': '#fc38db',
    'Black': '#000000',
    'White': '#ffffff'
}

# Experiment display names and colors
EXPERIMENT_CONFIG = {
    'mistral': {'name': 'Cloud VLM', 'color': PRESENTATION_COLORS['Blue']},
    'smol': {'name': 'Local VLM', 'color': PRESENTATION_COLORS['Cyan']},
    'novlm': {'name': 'No VLM', 'color': PRESENTATION_COLORS['Green']},
    'notrack': {'name': 'No Tracking', 'color': PRESENTATION_COLORS['Pink']}
}

# Scenario display names
SCENARIO_NAMES = {
    'frontal_approach': 'Frontal Approach',
    'intersection': 'Intersection',
    'doorway': 'Doorway'
}

# Constants
SYNC_TOLERANCE = 0.2  # seconds - max time diff for synchronization


def quaternion_to_yaw(q):
    """Convert quaternion to yaw angle."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class ScenarioWindow:
    """Time window for a single scenario run."""
    name: str           # e.g., "frontal_approach_human"
    start_time: float   # nanoseconds
    end_time: float     # nanoseconds (next scenario start or bag end)


class TrackingErrorExtractor:
    """Extract tracking error data from rosbag files."""
    
    def __init__(self, bag_path: str):
        self.bag_path = bag_path
        self.experiment_name = self._extract_experiment_name(bag_path)
        self.human_odom = []  # List of (timestamp_ns, Odometry)
        self.person_markers = []  # List of (timestamp_ns, MarkerArray)
        self.social_tasks = []  # List of (timestamp_ns, String)
    
    def _extract_experiment_name(self, bag_path: str) -> str:
        """Extract experiment name from bag path."""
        dirname = os.path.basename(bag_path.rstrip('/'))
        parts = dirname.split('_')
        if len(parts) >= 3:
            return parts[0]
        return dirname
    
    def read_bag(self):
        """Read all relevant messages from the bag file."""
        print(f"Reading bag: {self.bag_path}")
        
        # Find the .db3 file
        db3_file = None
        if os.path.isfile(self.bag_path) and self.bag_path.endswith('.db3'):
            db3_file = self.bag_path
            bag_dir = os.path.dirname(self.bag_path)
        else:
            bag_dir = self.bag_path
            for f in os.listdir(bag_dir):
                if f.endswith('.db3'):
                    db3_file = os.path.join(bag_dir, f)
                    break
        
        if not db3_file:
            raise FileNotFoundError(f"No .db3 file found in {self.bag_path}")
        
        # Check for metadata.yaml, generate if missing
        metadata_file = os.path.join(bag_dir, 'metadata.yaml')
        if not os.path.exists(metadata_file):
            print(f"  metadata.yaml missing, generating from .db3 schema...")
            self._generate_metadata(db3_file, metadata_file)
        
        # Setup reader
        storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Get topic types
        topic_types = {}
        for topic_info in reader.get_all_topics_and_types():
            topic_types[topic_info.name] = topic_info.type
        
        # Read messages
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            
            if topic == '/human/odom':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.human_odom.append((timestamp, msg))
                
            elif topic == '/social_costmap/person_markers':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.person_markers.append((timestamp, msg))
                
            elif topic == '/social_task':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.social_tasks.append((timestamp, msg))
        
        print(f"  Human odom: {len(self.human_odom)} msgs")
        print(f"  Person markers: {len(self.person_markers)} msgs")
        print(f"  Social tasks: {len(self.social_tasks)} msgs")
    
    def _generate_metadata(self, db3_path: str, metadata_path: str):
        """Generate metadata.yaml from .db3 file's internal schema."""
        import sqlite3
        import yaml
        
        conn = sqlite3.connect(db3_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, type, serialization_format FROM topics")
        topics_data = cursor.fetchall()
        
        topics = []
        for topic_id, name, msg_type, serialization_format in topics_data:
            topics.append({
                'name': name,
                'type': msg_type,
                'serialization_format': serialization_format,
                'offered_qos_profiles': ''
            })
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        message_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages")
        start_time, end_time = cursor.fetchone()
        
        conn.close()
        
        db3_filename = os.path.basename(db3_path)
        metadata = {
            'rosbag2_bagfile_information': {
                'version': 4,
                'storage_identifier': 'sqlite3',
                'relative_file_paths': [db3_filename],
                'duration': {
                    'nanoseconds': (end_time - start_time) if start_time and end_time else 0
                },
                'starting_time': {
                    'nanoseconds_since_epoch': start_time or 0
                },
                'message_count': message_count,
                'topics_with_message_count': [
                    {'topic_metadata': t, 'message_count': 0} for t in topics
                ],
                'compression_format': '',
                'compression_mode': ''
            }
        }
        
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        print(f"  Generated metadata.yaml with {len(topics)} topics")
    
    def segment_scenarios(self) -> list[ScenarioWindow]:
        """Segment the bag into individual scenario runs using /social_task."""
        if not self.social_tasks:
            print("Warning: No /social_task messages found. Cannot segment scenarios.")
            return []
        
        scenarios = []
        
        for i, (timestamp, msg) in enumerate(self.social_tasks):
            if i + 1 < len(self.social_tasks):
                end_time = self.social_tasks[i + 1][0]
            else:
                all_times = []
                if self.human_odom:
                    all_times.append(self.human_odom[-1][0])
                if self.person_markers:
                    all_times.append(self.person_markers[-1][0])
                end_time = max(all_times) if all_times else timestamp + 60_000_000_000
            
            scenarios.append(ScenarioWindow(
                name=msg.data,
                start_time=timestamp,
                end_time=end_time
            ))
        
        print(f"Found {len(scenarios)} scenario runs")
        return scenarios
    
    def _filter_by_window(self, messages: list, window: ScenarioWindow) -> list:
        """Filter messages to those within the scenario window."""
        return [(t, m) for t, m in messages if window.start_time <= t < window.end_time]
    
    def compute_tracking_errors(self) -> list[dict]:
        """Compute tracking errors for all scenarios."""
        self.read_bag()
        scenarios = self.segment_scenarios()
        
        all_errors = []
        
        for window in scenarios:
            # Parse scenario name
            scenario_name = window.name.replace('_human', '')
            scenario_display = SCENARIO_NAMES.get(scenario_name, scenario_name)
            
            # Filter messages for this window
            human_odom_window = self._filter_by_window(self.human_odom, window)
            person_markers_window = self._filter_by_window(self.person_markers, window)
            
            # Extract positions as NumPy arrays
            human_timestamps = []
            human_positions = []
            for timestamp, odom in human_odom_window:
                x = odom.pose.pose.position.x
                y = odom.pose.pose.position.y
                human_timestamps.append(timestamp)
                human_positions.append((x, y))
            human_timestamps = np.array(human_timestamps, dtype=np.int64)
            human_positions = np.array(human_positions, dtype=np.float64)
            
            tracked_timestamps = []
            tracked_positions = []
            for timestamp, markers in person_markers_window:
                for marker in markers.markers:
                    if marker.ns == 'detected_persons':
                        x = marker.pose.position.x
                        y = marker.pose.position.y
                        if abs(x) > 0.01 or abs(y) > 0.01:
                            tracked_timestamps.append(timestamp)
                            tracked_positions.append((x, y))
                        break
            
            tracked_timestamps = np.array(tracked_timestamps, dtype=np.int64)
            tracked_positions = np.array(tracked_positions, dtype=np.float64)
            
            # Match positions and compute errors using binary search
            if len(tracked_timestamps) == 0:
                continue
            
            for i, h_ts in enumerate(human_timestamps):
                # Binary search for closest tracked position
                idx = bisect.bisect_left(tracked_timestamps, h_ts)
                
                # Determine which neighbor is closer
                if idx == 0:
                    closest_idx = 0
                elif idx >= len(tracked_timestamps):
                    closest_idx = len(tracked_timestamps) - 1
                elif (tracked_timestamps[idx] - h_ts) < (h_ts - tracked_timestamps[idx - 1]):
                    closest_idx = idx
                else:
                    closest_idx = idx - 1
                
                # Check if within tolerance
                if abs(tracked_timestamps[closest_idx] - h_ts) <= SYNC_TOLERANCE * 1e9:
                    h_x, h_y = human_positions[i]
                    t_x, t_y = tracked_positions[closest_idx]
                    error = math.sqrt((h_x - t_x)**2 + (h_y - t_y)**2)
                    all_errors.append({
                        'experiment': self.experiment_name,
                        'scenario': scenario_display,
                        'tracking_error': error
                    })
        
        return all_errors


def process_single_bag_task4(bag_path: str) -> list[dict]:
    """Process a single bag file - for multiprocessing."""
    try:
        print(f"Processing: {os.path.basename(bag_path)}")
        extractor = TrackingErrorExtractor(bag_path)
        errors = extractor.compute_tracking_errors()
        print(f"  Extracted {len(errors)} tracking error samples from {os.path.basename(bag_path)}")
        return errors
    except Exception as e:
        print(f"  Error processing {bag_path}: {e}")
        return []


def find_bags_in_directory(bag_dir: str):
    """Find all rosbag directories in the given directory."""
    bags = []
    for entry in os.listdir(bag_dir):
        entry_path = os.path.join(bag_dir, entry)
        if os.path.isdir(entry_path):
            for f in os.listdir(entry_path):
                if f.endswith('.db3'):
                    bags.append(entry_path)
                    break
    return sorted(bags)


def plot_tracking_performance(errors_df: pd.DataFrame, output_path: str):
    """Create violin plot of tracking errors per scenario for No VLM only."""
    if errors_df.empty:
        print("No tracking error data found!")
        return
    
    # Map experiment names
    errors_df['experiment_display'] = errors_df['experiment'].map(
        lambda x: EXPERIMENT_CONFIG.get(x, {}).get('name', x)
    )
    
    # Filter for No VLM experiments only
    errors_df = errors_df[errors_df['experiment'] == 'novlm']
    
    if errors_df.empty:
        print("No No VLM tracking error data found!")
        return
    
    # Set seaborn style
    sns.set_style('whitegrid')
    
    # Create figure with subplots for each scenario
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    for i, scenario in enumerate(['Frontal Approach', 'Intersection', 'Doorway']):
        ax = axes[i]
        
        # Filter for this scenario
        scenario_df = errors_df[errors_df['scenario'] == scenario]
        
        if scenario_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(scenario, fontsize=12, fontweight='bold')
            continue
        
        # Create violin plot using seaborn
        sns.violinplot(data=scenario_df, y='tracking_error',
                       palette=[PRESENTATION_COLORS['Green']],
                       inner='box', saturation=0.7, ax=ax,
                       cut=0, bw_adjust=0.5)
        
        # Set alpha for violin bodies
        for pc in ax.collections:
            pc.set_alpha(0.7)
        
        ax.set_xlabel('', fontsize=11)
        ax.set_ylabel('Tracking Error (m)', fontsize=11)
        ax.set_title(scenario, fontsize=12, fontweight='bold')
        ax.set_xticklabels([])
        ax.set_yscale('log')
        ax.set_ylim(0.01, 15)
        ax.grid(axis='y', alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Print statistics
    print(f"\nTracking Error Statistics (No VLM only):")
    for scenario in ['Frontal Approach', 'Intersection', 'Doorway']:
        scenario_df = errors_df[errors_df['scenario'] == scenario]
        if not scenario_df.empty:
            errors = scenario_df['tracking_error'].values
            print(f"\n{scenario}:")
            print(f"  No VLM: mean={np.mean(errors):.4f}m, std={np.std(errors):.4f}m, n={len(errors)}")


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bag_dir = os.path.join(script_dir, '../../src/tests/rosbags')
    output_path = os.path.join(script_dir, 'tracking_performance.svg')
    
    # Find all bag files
    print(f"Searching for rosbags in {bag_dir}...")
    bag_paths = find_bags_in_directory(bag_dir)
    print(f"Found {len(bag_paths)} bag files")
    
    # Process bags in parallel
    num_workers = min(cpu_count() - 2, len(bag_paths), 8)
    print(f"Processing with {num_workers} parallel workers...\n")
    
    with Pool(num_workers) as pool:
        results = pool.map(process_single_bag_task4, bag_paths)
    
    # Flatten results
    all_errors = [e for bag_errors in results for e in bag_errors]
    
    # Convert to DataFrame
    errors_df = pd.DataFrame(all_errors)
    
    # Generate plot
    print(f"\nGenerating tracking performance violin plot...")
    plot_tracking_performance(errors_df, output_path)
    
    print(f"\nDone! Output saved to: {output_path}")


if __name__ == '__main__':
    main()
