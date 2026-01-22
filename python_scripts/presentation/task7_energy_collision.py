#!/usr/bin/env python3
"""
Task 7: Violin Plot of Robot Velocity at Collision

Creates a violin plot showing robot velocity at collision per experiment.
Identifies collision events (min_distance < 0), extracts robot velocity
from /mobile_base_controller/odom at collision time.
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
HUMAN_RADIUS = 0.25  # meters
ROBOT_HALF_LENGTH = 0.35  # meters
ROBOT_HALF_WIDTH = 0.24  # meters
SYNC_TOLERANCE = 0.2  # seconds


def quaternion_to_yaw(q):
    """Convert quaternion to yaw angle."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def compute_edge_distance(robot_x: float, robot_y: float, robot_yaw: float,
                        human_x: float, human_y: float) -> float:
    """Compute minimum edge-to-edge distance between robot rectangle and human circle."""
    dx = human_x - robot_x
    dy = human_y - robot_y
    cos_yaw = math.cos(-robot_yaw)
    sin_yaw = math.sin(-robot_yaw)
    local_x = dx * cos_yaw - dy * sin_yaw
    local_y = dx * sin_yaw + dy * cos_yaw
    
    closest_x = max(-ROBOT_HALF_LENGTH, min(ROBOT_HALF_LENGTH, local_x))
    closest_y = max(-ROBOT_HALF_WIDTH, min(ROBOT_HALF_WIDTH, local_y))
    
    dist_to_human_center = math.sqrt((local_x - closest_x)**2 + (local_y - closest_y)**2)
    edge_distance = dist_to_human_center - HUMAN_RADIUS
    
    return edge_distance


@dataclass
class ScenarioWindow:
    """Time window for a single scenario run."""
    name: str
    start_time: float
    end_time: float


class CollisionVelocityExtractor:
    """Extract robot velocity at collision data from rosbag files."""
    
    def __init__(self, bag_path: str):
        self.bag_path = bag_path
        self.experiment_name = self._extract_experiment_name(bag_path)
        self.robot_odom = []  # List of (timestamp_ns, Odometry)
        self.human_odom = []  # List of (timestamp_ns, Odometry)
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
            
            if topic == '/mobile_base_controller/odom':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.robot_odom.append((timestamp, msg))
                
            elif topic == '/human/odom':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.human_odom.append((timestamp, msg))
                
            elif topic == '/social_task':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.social_tasks.append((timestamp, msg))
        
        print(f"  Robot odom: {len(self.robot_odom)} msgs")
        print(f"  Human odom: {len(self.human_odom)} msgs")
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
            return []
        
        scenarios = []
        
        for i, (timestamp, msg) in enumerate(self.social_tasks):
            if i + 1 < len(self.social_tasks):
                end_time = self.social_tasks[i + 1][0]
            else:
                all_times = []
                if self.robot_odom:
                    all_times.append(self.robot_odom[-1][0])
                if self.human_odom:
                    all_times.append(self.human_odom[-1][0])
                end_time = max(all_times) if all_times else timestamp + 60_000_000_000
            
            scenarios.append(ScenarioWindow(
                name=msg.data,
                start_time=timestamp,
                end_time=end_time
            ))
        
        return scenarios
    
    def _filter_by_window(self, messages: list, window: ScenarioWindow) -> list:
        """Filter messages to those within the scenario window."""
        return [(t, m) for t, m in messages if window.start_time <= t < window.end_time]
    
    def compute_collision_velocities(self) -> list[dict]:
        """Compute robot velocities at collision for all scenarios."""
        self.read_bag()
        scenarios = self.segment_scenarios()
        
        all_velocities = []
        
        for window in scenarios:
            scenario_name = window.name.replace('_human', '')
            scenario_display = SCENARIO_NAMES.get(scenario_name, scenario_name)
            
            robot_odom_window = self._filter_by_window(self.robot_odom, window)
            human_odom_window = self._filter_by_window(self.human_odom, window)
            
            # Early continue for empty windows
            if not robot_odom_window or not human_odom_window:
                continue
            
            # Pre-compute robot timestamps and poses as NumPy arrays for fast access
            robot_timestamps = np.array([t for t, _ in robot_odom_window], dtype=np.int64)
            robot_poses = []
            for timestamp, odom in robot_odom_window:
                q = odom.pose.pose.orientation
                yaw = quaternion_to_yaw(q)
                vx = odom.twist.twist.linear.x
                vy = odom.twist.twist.linear.y
                v_mag = math.sqrt(vx**2 + vy**2)
                robot_poses.append((odom.pose.pose.position.x,
                                   odom.pose.pose.position.y, yaw, v_mag))
            robot_poses = np.array(robot_poses, dtype=np.float64)
            
            # Track collision state to avoid duplicate entries
            in_collision = False
            
            # Process human poses and find collisions
            for timestamp, odom in human_odom_window:
                h_x = odom.pose.pose.position.x
                h_y = odom.pose.pose.position.y
                
                # Binary search for closest robot timestamp
                idx = bisect.bisect_left(robot_timestamps, timestamp)
                
                # Determine which neighbor is closer
                if idx == 0:
                    closest_idx = 0
                elif idx >= len(robot_timestamps):
                    closest_idx = len(robot_timestamps) - 1
                elif (robot_timestamps[idx] - timestamp) < (timestamp - robot_timestamps[idx - 1]):
                    closest_idx = idx
                else:
                    closest_idx = idx - 1
                
                # Check sync tolerance
                if abs(robot_timestamps[closest_idx] - timestamp) > SYNC_TOLERANCE * 1e9:
                    in_collision = False
                    continue
                
                r_x, r_y, r_yaw, r_vmag = robot_poses[closest_idx]
                
                # Check for collision
                edge_dist = compute_edge_distance(r_x, r_y, r_yaw, h_x, h_y)
                
                if edge_dist < 0:  # Collision detected
                    if not in_collision:  # Only record first frame of collision
                        all_velocities.append({
                            'experiment': self.experiment_name,
                            'scenario': scenario_display,
                            'robot_velocity': r_vmag
                        })
                        in_collision = True
                else:
                    in_collision = False
        
        return all_velocities


def process_single_bag(bag_path: str) -> list[dict]:
    """Process a single bag file - for multiprocessing."""
    try:
        print(f"Processing: {os.path.basename(bag_path)}")
        extractor = CollisionVelocityExtractor(bag_path)
        velocities = extractor.compute_collision_velocities()
        print(f"  Found {len(velocities)} collision events in {os.path.basename(bag_path)}")
        return velocities
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


def plot_robot_velocity(velocities_df: pd.DataFrame, output_path: str):
    """Create violin plot of robot velocity at collision per experiment."""
    if velocities_df.empty:
        print("No robot velocity data found!")
        return
    
    # Map experiment names
    velocities_df['experiment_display'] = velocities_df['experiment'].map(
        lambda x: EXPERIMENT_CONFIG.get(x, {}).get('name', x)
    )
    
    # Set seaborn style
    sns.set_style('whitegrid')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter data to only include experiments in the desired order
    experiments = ['No VLM', 'Local VLM', 'Cloud VLM', 'No Tracking']
    df_filtered = velocities_df[velocities_df['experiment_display'].isin(experiments)].copy()
    
    # Create color palette in order
    color_palette = []
    for exp_name in experiments:
        for exp_key, config in EXPERIMENT_CONFIG.items():
            if config['name'] == exp_name:
                color_palette.append(config['color'])
                break
    
    # Create violin plot using seaborn
    sns.violinplot(data=df_filtered, x='experiment_display', y='robot_velocity',
                   order=experiments, palette=color_palette,
                   inner='box', saturation=0.7, ax=ax,
                   cut=0, bw_adjust=0.5)
    
    # Set alpha for violin bodies
    for pc in ax.collections:
        pc.set_alpha(0.7)
    
    # Set labels and title
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Robot Velocity at Collision (m/s)', fontsize=12)
    ax.set_title('Robot Velocity at Collision by Experiment', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Print statistics
    print(f"\nRobot Velocity at Collision Statistics:")
    for exp in experiments:
        exp_df = velocities_df[velocities_df['experiment_display'] == exp]
        if not exp_df.empty:
            velocities = exp_df['robot_velocity'].values
            print(f"\n{exp}:")
            print(f"  Collisions: {len(velocities)}")
            print(f"  Velocity: mean={np.mean(velocities):.3f}m/s, std={np.std(velocities):.3f}m/s")
            print(f"  Range: {np.min(velocities):.3f}m/s - {np.max(velocities):.3f}m/s")


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bag_dir = os.path.join(script_dir, '../../src/tests/rosbags')
    output_path = os.path.join(script_dir, 'energy_collision.svg')
    
    # Find all bag files
    print(f"Searching for rosbags in {bag_dir}...")
    bag_paths = find_bags_in_directory(bag_dir)
    print(f"Found {len(bag_paths)} bag files")
    
    # Process bags in parallel
    num_workers = min(cpu_count() - 2, len(bag_paths), 8)  # Use most cores but keep some free
    print(f"Processing with {num_workers} parallel workers...\n")
    
    with Pool(num_workers) as pool:
        results = pool.map(process_single_bag, bag_paths)
    
    # Flatten results
    all_velocities = [v for bag_velocities in results for v in bag_velocities]
    
    # Convert to DataFrame
    velocities_df = pd.DataFrame(all_velocities)
    
    # Generate plot
    print(f"\nGenerating robot velocity at collision violin plot...")
    plot_robot_velocity(velocities_df, output_path)
    
    print(f"\nDone! Output saved to: {output_path}")


if __name__ == '__main__':
    main()
