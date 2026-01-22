#!/usr/bin/env python3
"""
Task 3: Velocity Estimation Accuracy (Scatter Plot)

Creates a scatter plot comparing predicted vs. actual human velocities.
Extracts human velocity from /human/odom (actual) and tracked velocity from
/social_costmap/person_markers (predicted).
"""

import os
import sys
import math
import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import yaml
from multiprocessing import Pool, cpu_count

# ROS2 imports - must be run in ROS2 environment
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from nav_msgs.msg import Odometry
    from visualization_msgs.msg import MarkerArray
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

# Constants
SYNC_TOLERANCE = 0.2  # seconds - max time diff for synchronization


def quaternion_to_yaw(q):
    """Convert quaternion to yaw angle."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class VelocityExtractor:
    """Extract velocity data from rosbag files."""
    
    def __init__(self, bag_path: str):
        self.bag_path = bag_path
        self.human_odom = []  # List of (timestamp_ns, Odometry)
        self.person_markers = []  # List of (timestamp_ns, MarkerArray)
    
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
            print(f"  No .db3 file found in {self.bag_path}")
            return
        
        # Check for metadata.yaml, generate if missing
        metadata_file = os.path.join(bag_dir, 'metadata.yaml')
        if not os.path.exists(metadata_file):
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
        
        print(f"  Human odom: {len(self.human_odom)} msgs")
        print(f"  Person markers: {len(self.person_markers)} msgs")
    
    def _generate_metadata(self, db3_path: str, metadata_path: str):
        """Generate metadata.yaml from .db3 file's internal schema."""
        
        conn = sqlite3.connect(db3_path)
        cursor = conn.cursor()
        
        # Get topics from the topics table
        cursor.execute("SELECT id, name, type, serialization_format FROM topics")
        topics_data = cursor.fetchall()
        
        topics = []
        for topic_id, name, msg_type, serialization_format in topics_data:
            topics.append({
                'name': name,
                'type': msg_type,
                'serialization_format': serialization_format or 'cdr',
                'offered_qos_profiles': ''
            })
        
        # Get message count
        cursor.execute("SELECT COUNT(*) FROM messages")
        message_count = cursor.fetchone()[0]
        
        # Get time range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages")
        start_time, end_time = cursor.fetchone()
        
        conn.close()
        
        # Build metadata
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
    
    def extract_velocity_data(self):
        """
        Extract matched actual vs. predicted velocities.
        Uses velocity arrow markers which encode velocity directly.
        
        Arrow encoding:
          - scale.x = speed * PREDICTION_TIME (2.0 seconds)
          - orientation (z, w) = (sin(heading/2), cos(heading/2))
        
        Returns: List of (actual_vx, actual_vy, actual_vmag, predicted_vx, predicted_vy, predicted_vmag)
        """
        velocity_data = []
        
        # Convert human odom to velocity data as NumPy arrays
        human_timestamps = []
        human_velocities = []
        for timestamp, odom in self.human_odom:
            vx = odom.twist.twist.linear.x
            vy = odom.twist.twist.linear.y
            vmag = math.sqrt(vx**2 + vy**2)
            human_timestamps.append(timestamp)
            human_velocities.append((vx, vy, vmag))
        
        human_timestamps = np.array(human_timestamps, dtype=np.int64)
        human_velocities = np.array(human_velocities, dtype=np.float64)
        
        # DEBUG: Print human velocity info
        if human_velocities.size > 0:
            unique_vmag = set(round(v[2], 3) for v in human_velocities)
            print(f"  DEBUG: Human velocity unique values: {sorted(unique_vmag)}")
        
        # Extract velocity from arrow markers (namespace = 'detected_persons_velocity')
        # Arrow scale.x = speed * PREDICTION_TIME
        # Arrow orientation encodes velocity direction
        PREDICTION_TIME = 2.0  # Must match social_costmap_publisher.py
        
        marker_timestamps = []
        marker_velocities = []
        
        for timestamp, markers in self.person_markers:
            for marker in markers.markers:
                # Look for velocity arrows
                if marker.ns == 'detected_persons_velocity':
                    # Decode velocity from arrow
                    arrow_length = marker.scale.x  # = speed * PREDICTION_TIME
                    speed = arrow_length / PREDICTION_TIME
                    
                    # Decode velocity direction from quaternion
                    # arrow.pose.orientation = (x=0, y=0, z=sin(heading/2), w=cos(heading/2))
                    qz = marker.pose.orientation.z
                    qw = marker.pose.orientation.w
                    heading = 2.0 * math.atan2(qz, qw)
                    
                    # Recover velocity components
                    vx = speed * math.cos(heading)
                    vy = speed * math.sin(heading)
                    vmag = math.sqrt(vx**2 + vy**2)
                    
                    marker_timestamps.append(timestamp)
                    marker_velocities.append((vx, vy, vmag))
                    break  # Only take first velocity arrow per message
        
        print(f"  DEBUG: Velocity arrows found: {len(marker_velocities)}")
        
        if marker_velocities:
            mags = [v[2] for v in marker_velocities]
            print(f"  DEBUG: Predicted velocity stats: min={min(mags):.4f}, max={max(mags):.4f}, mean={np.mean(mags):.4f} m/s")
            # Show sample velocities
            print(f"  DEBUG: Sample velocities (first 3):")
            for i in range(min(3, len(marker_velocities))):
                print(f"    [{i}] vx={marker_velocities[i][0]:.4f}, vy={marker_velocities[i][1]:.4f}, vmag={marker_velocities[i][2]:.4f}")
        
        # Convert to NumPy arrays
        marker_timestamps = np.array(marker_timestamps, dtype=np.int64)
        marker_velocities = np.array(marker_velocities, dtype=np.float64) if marker_velocities else np.array([]).reshape(0, 3)
        
        # Match velocities by timestamp
        if len(marker_timestamps) == 0:
            print(f"  WARNING: No velocity arrows found!")
            return velocity_data
        
        tolerance_ns = int(SYNC_TOLERANCE * 1e9)
        
        for i, h_ts in enumerate(human_timestamps):
            idx = np.searchsorted(marker_timestamps, h_ts)
            
            best_idx = None
            best_diff = tolerance_ns + 1
            
            for check_idx in [idx - 1, idx, idx + 1]:
                if 0 <= check_idx < len(marker_timestamps):
                    diff = abs(marker_timestamps[check_idx] - h_ts)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = check_idx
            
            if best_idx is not None and best_diff <= tolerance_ns:
                hv = human_velocities[i]
                mv = marker_velocities[best_idx]
                velocity_data.append((hv[0], hv[1], hv[2], mv[0], mv[1], mv[2]))
        
        return velocity_data


def find_bags_in_directory(bag_dir: str, prefix_filter: str = None):
    """Find all rosbag directories in the given directory."""
    bags = []
    for entry in os.listdir(bag_dir):
        entry_path = os.path.join(bag_dir, entry)
        if os.path.isdir(entry_path):
            # Check if it contains a .db3 file
            has_db3 = any(f.endswith('.db3') for f in os.listdir(entry_path))
            if has_db3:
                # Apply prefix filter if specified
                if prefix_filter is None or entry.startswith(prefix_filter):
                    bags.append(entry_path)
    return sorted(bags)


def plot_velocity_accuracy(velocity_data: list, output_path: str):
    """Create scatter plot of predicted vs. actual velocity magnitudes."""
    if not velocity_data:
        print("No velocity data to plot!")
        return
    
    # Extract data - only magnitude
    actual_vmag = [d[2] for d in velocity_data]
    pred_vmag = [d[5] for d in velocity_data]
    
    # Debug output
    print(f"\n  DEBUG: Final velocity data samples: {len(velocity_data)}")
    print(f"  DEBUG: Actual velocity range: min={np.min(actual_vmag):.4f} m/s, max={np.max(actual_vmag):.4f} m/s, mean={np.mean(actual_vmag):.4f} m/s")
    print(f"  DEBUG: Predicted velocity range: min={np.min(pred_vmag):.4f} m/s, max={np.max(pred_vmag):.4f} m/s, mean={np.mean(pred_vmag):.4f} m/s")
    
    # Compute correlation
    correlation = np.corrcoef(actual_vmag, pred_vmag)[0, 1]
    print(f"  DEBUG: Correlation coefficient: {correlation:.4f}")
    
    print(f"  DEBUG: Sample pairs (first 10):")
    for i in range(min(10, len(velocity_data))):
        print(f"    [{i}] actual=({velocity_data[i][0]:.3f}, {velocity_data[i][1]:.3f}) |{velocity_data[i][2]:.3f}| m/s, "
              f"predicted=({velocity_data[i][3]:.3f}, {velocity_data[i][4]:.3f}) |{velocity_data[i][5]:.3f}| m/s")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot magnitude only
    ax.scatter(actual_vmag, pred_vmag, color=PRESENTATION_COLORS['Green'], 
               alpha=0.6, edgecolors='none', s=50)
    
    # Diagonal line (perfect prediction) - use pink color
    min_val = min(min(actual_vmag), min(pred_vmag))
    max_val = max(max(actual_vmag), max(pred_vmag))
    ax.plot([min_val, max_val], [min_val, max_val], color=PRESENTATION_COLORS['Pink'], 
            linestyle='--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Velocity Magnitude (m/s)', fontsize=12)
    ax.set_ylabel('Predicted Velocity Magnitude (m/s)', fontsize=12)
    ax.set_title(f'Velocity Estimation Accuracy (r={correlation:.3f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Print statistics
    print(f"\nVelocity Accuracy Statistics:")
    print(f"  Total matched samples: {len(velocity_data)}")
    
    # Calculate errors for magnitude only
    vmag_errors = [abs(a - p) for a, p in zip(actual_vmag, pred_vmag)]
    
    print(f"  |V| error: mean={np.mean(vmag_errors):.4f} m/s, std={np.std(vmag_errors):.4f} m/s")


def process_single_bag_task3(bag_path: str) -> list:
    """Process a single bag file - for multiprocessing."""
    try:
        print(f"Processing: {os.path.basename(bag_path)}")
        extractor = VelocityExtractor(bag_path)
        extractor.read_bag()
        velocity_data = extractor.extract_velocity_data()
        print(f"  Extracted {len(velocity_data)} velocity samples from {os.path.basename(bag_path)}")
        return velocity_data
    except Exception as e:
        print(f"  Error processing {bag_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bag_dir = os.path.join(script_dir, '../../src/tests/rosbags')
    output_path = os.path.join(script_dir, 'velocity_accuracy.png')
    
    # Only use novlm bags
    print(f"Searching for novlm rosbags in {bag_dir}...")
    bag_paths = find_bags_in_directory(bag_dir, prefix_filter='novlm_')
    print(f"Found {len(bag_paths)} novlm bag files")
    
    if not bag_paths:
        print("No bags found!")
        return
    
    # Process bags in parallel
    num_workers = min(cpu_count() - 2, len(bag_paths), 8)
    num_workers = max(1, num_workers)
    print(f"Processing with {num_workers} parallel workers...\n")
    
    with Pool(num_workers) as pool:
        results = pool.map(process_single_bag_task3, bag_paths)
    
    # Flatten results
    all_velocity_data = [v for bag_velocities in results for v in bag_velocities]
    
    # Generate plot
    print(f"\nGenerating velocity accuracy scatter plot...")
    plot_velocity_accuracy(all_velocity_data, output_path)
    
    print(f"\nDone! Output saved to: {output_path}")


if __name__ == '__main__':
    main()
