#!/usr/bin/env python3
"""
Rosbag Metrics Extraction Tool

Extracts 6 social navigation metrics from ROS2 bag recordings:
1. PSC - Personal Space Compliance (% timesteps >= 1.0m)
2. TTC - Minimum Time-to-Collision
3. MinDist - Minimum distance to human
4. DetLat - Detection latency (human spawn -> first detection)
5. VLMLat - VLM response latency
6. Duration - Scenario total time

Usage:
    python extract_metrics.py --bag <bag_path> --output <csv_path>
    python extract_metrics.py --bag-dir <dir> --output <csv_path>
"""

import argparse
import os
import sys
import math
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np

# ROS2 imports - must be run in ROS2 environment
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import TransformStamped
except ImportError as e:
    print(f"Error: ROS2 packages not found. Run this script in a ROS2 environment.")
    print(f"Details: {e}")
    sys.exit(1)

from metrics_types import ScenarioMetrics, TimestampedPose, write_metrics_csv


# Constants
PSC_THRESHOLD = 1.25  # meters - personal space threshold (edge-to-edge)
SYNC_TOLERANCE = 0.2  # seconds - max time diff for pose synchronization
ROBOT_BASE_FRAME = 'base_footprint'
MAP_FRAME = 'map'

# Footprint definitions
HUMAN_RADIUS = 0.25  # meters - human as circle
# Robot footprint: [[0.35, 0.24], [-0.35, 0.24], [-0.35, -0.24], [0.35, -0.24]]
# This is a rectangle 0.7m (front-back) x 0.48m (side-to-side)
ROBOT_HALF_LENGTH = 0.35  # meters - front/back from center
ROBOT_HALF_WIDTH = 0.24   # meters - left/right from center


def compute_edge_distance(robot_x: float, robot_y: float, robot_yaw: float,
                          human_x: float, human_y: float) -> float:
    """
    Compute minimum edge-to-edge distance between robot rectangle and human circle.
    
    Robot: rectangle centered at (robot_x, robot_y) with orientation robot_yaw
    Human: circle centered at (human_x, human_y) with radius HUMAN_RADIUS
    
    Returns: distance from robot edge to human edge (negative if overlapping)
    """
    # Transform human position to robot's local frame
    dx = human_x - robot_x
    dy = human_y - robot_y
    cos_yaw = math.cos(-robot_yaw)
    sin_yaw = math.sin(-robot_yaw)
    local_x = dx * cos_yaw - dy * sin_yaw
    local_y = dx * sin_yaw + dy * cos_yaw
    
    # Find closest point on robot rectangle to human center
    closest_x = max(-ROBOT_HALF_LENGTH, min(ROBOT_HALF_LENGTH, local_x))
    closest_y = max(-ROBOT_HALF_WIDTH, min(ROBOT_HALF_WIDTH, local_y))
    
    # Distance from closest point on robot to human center
    dist_to_human_center = math.sqrt((local_x - closest_x)**2 + (local_y - closest_y)**2)
    
    # Edge-to-edge distance (subtract human radius)
    edge_distance = dist_to_human_center - HUMAN_RADIUS
    
    return edge_distance


@dataclass
class ScenarioWindow:
    """Time window for a single scenario run."""
    name: str           # e.g., "frontal_approach_human"
    start_time: float   # nanoseconds
    end_time: float     # nanoseconds (next scenario start or bag end)


def quaternion_to_yaw(q):
    """Convert quaternion to yaw angle."""
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RosbagMetricsExtractor:
    """Extract social navigation metrics from a ROS2 bag file."""
    
    def __init__(self, bag_path: str):
        self.bag_path = bag_path
        self.experiment_name = self._extract_experiment_name(bag_path)
        
        # Message storage
        self.robot_poses = []     # List of (timestamp_ns, x, y, vx, vy) from TF
        self.human_odom = []      # List of (timestamp_ns, Odometry)
        self.social_tasks = []    # List of (timestamp_ns, String)
        self.person_markers = []  # List of (timestamp_ns, MarkerArray)
        self.vlm_responses = []   # List of (timestamp_ns, String)
        
        # TF buffer for computing robot poses
        self.tf_buffer = {}  # frame_id -> list of (timestamp, transform)
        
    def _extract_experiment_name(self, bag_path: str) -> str:
        """Extract experiment name from bag path (e.g., 'mistral' from 'mistral_20251211_172308')."""
        dirname = os.path.basename(bag_path.rstrip('/'))
        # Format: experimentname_YYYYMMDD_HHMMSS
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
        
        # Collect TF transforms and robot odom messages
        tf_transforms = []  # (timestamp, transform)
        robot_odom_msgs = []  # (timestamp, Odometry) - if available
        
        # Read messages
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            
            if topic == '/tf':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                for transform in msg.transforms:
                    tf_transforms.append((timestamp, transform))
            
            elif topic == '/mobile_base_controller/odom':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                robot_odom_msgs.append((timestamp, msg))
                    
            elif topic == '/human/odom':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.human_odom.append((timestamp, msg))
                
            elif topic == '/social_task':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.social_tasks.append((timestamp, msg))
                
            elif topic == '/social_costmap/person_markers':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.person_markers.append((timestamp, msg))
                
            elif topic == '/vlm/response':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.vlm_responses.append((timestamp, msg))
        
        # Extract robot poses: prefer direct odom if available, else use TF
        if robot_odom_msgs:
            print(f"  Using direct robot odom: {len(robot_odom_msgs)} msgs")
            self.robot_poses = [self._odom_to_pose(t, m) for t, m in robot_odom_msgs]
        else:
            self.robot_poses = self._extract_robot_poses_from_tf(tf_transforms)
            print(f"  Robot poses (from TF): {len(self.robot_poses)}")
        
        print(f"  Human odom: {len(self.human_odom)} msgs")
        print(f"  Social tasks: {len(self.social_tasks)} msgs")
        print(f"  Person markers: {len(self.person_markers)} msgs")
        print(f"  VLM responses: {len(self.vlm_responses)} msgs")
    
    def _extract_robot_poses_from_tf(self, tf_transforms: List[Tuple]) -> List[TimestampedPose]:
        """
        Extract robot poses from TF transforms.
        
        Looks for map -> base_footprint transforms (or via odom frame).
        """
        # Build TF tree - we need map -> odom -> base_footprint chain
        # Store transforms indexed by (parent, child) -> list of (timestamp, transform)
        transforms = defaultdict(list)
        
        for timestamp, tf in tf_transforms:
            key = (tf.header.frame_id, tf.child_frame_id)
            transforms[key].append((timestamp, tf))
        
        print(f"  TF frames found: {list(transforms.keys())[:10]}...")
        
        poses = []
        
        # Look for direct map -> base_footprint or map -> base_link
        for base_frame in ['base_footprint', 'base_link']:
            key = (MAP_FRAME, base_frame)
            if key in transforms:
                print(f"  Found direct {MAP_FRAME} -> {base_frame} transforms: {len(transforms[key])}")
                for timestamp, tf in transforms[key]:
                    yaw = quaternion_to_yaw(tf.transform.rotation)
                    poses.append(TimestampedPose(
                        timestamp=timestamp / 1e9,
                        x=tf.transform.translation.x,
                        y=tf.transform.translation.y,
                        yaw=yaw,
                        vx=0.0,  # TF doesn't include velocity
                        vy=0.0
                    ))
                break
        
        # If not found, try to compose map -> odom -> base_footprint
        if not poses:
            map_to_odom = transforms.get((MAP_FRAME, 'odom'), [])
            odom_to_base = transforms.get(('odom', 'base_footprint'), transforms.get(('odom', 'base_link'), []))
            
            if map_to_odom and odom_to_base:
                print(f"  Composing {MAP_FRAME} -> odom ({len(map_to_odom)}) -> base ({len(odom_to_base)})")
                poses = self._compose_transforms(map_to_odom, odom_to_base)
        
        # Estimate velocities from position differences
        if len(poses) >= 2:
            poses = self._estimate_velocities(poses)
        
        return poses
    
    def _compose_transforms(self, parent_transforms: List, child_transforms: List) -> List[TimestampedPose]:
        """Compose two transform chains to get final poses."""
        poses = []
        
        # Sort by timestamp
        parent_transforms.sort(key=lambda x: x[0])
        child_transforms.sort(key=lambda x: x[0])
        
        parent_idx = 0
        
        for child_ts, child_tf in child_transforms:
            # Find closest parent transform
            while (parent_idx < len(parent_transforms) - 1 and 
                   parent_transforms[parent_idx + 1][0] <= child_ts):
                parent_idx += 1
            
            if parent_idx < len(parent_transforms):
                parent_ts, parent_tf = parent_transforms[parent_idx]
                
                # Simple composition: just add translations (ignoring rotation for now)
                # This is a simplification - full composition would need quaternion math
                parent_yaw = quaternion_to_yaw(parent_tf.transform.rotation)
                
                # Transform child position by parent rotation and translation
                cx, cy = child_tf.transform.translation.x, child_tf.transform.translation.y
                rotated_x = cx * math.cos(parent_yaw) - cy * math.sin(parent_yaw)
                rotated_y = cx * math.sin(parent_yaw) + cy * math.cos(parent_yaw)
                
                final_x = parent_tf.transform.translation.x + rotated_x
                final_y = parent_tf.transform.translation.y + rotated_y
                
                # Compose yaw: add parent yaw to child yaw
                child_yaw = quaternion_to_yaw(child_tf.transform.rotation)
                final_yaw = parent_yaw + child_yaw
                
                poses.append(TimestampedPose(
                    timestamp=child_ts / 1e9,
                    x=final_x,
                    y=final_y,
                    yaw=final_yaw,
                    vx=0.0,
                    vy=0.0
                ))
        
        return poses
    
    def _estimate_velocities(self, poses: List[TimestampedPose]) -> List[TimestampedPose]:
        """Estimate velocities from position differences."""
        for i in range(1, len(poses)):
            dt = poses[i].timestamp - poses[i-1].timestamp
            if dt > 0.001:  # Avoid division by very small dt
                poses[i].vx = (poses[i].x - poses[i-1].x) / dt
                poses[i].vy = (poses[i].y - poses[i-1].y) / dt
        
        # First pose has same velocity as second
        if len(poses) >= 2:
            poses[0].vx = poses[1].vx
            poses[0].vy = poses[1].vy
        
        return poses
    
    def _generate_metadata(self, db3_path: str, metadata_path: str):
        """Generate metadata.yaml from .db3 file's internal schema."""
        import sqlite3
        import yaml
        
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
                'serialization_format': serialization_format,
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
        
        print(f"  Generated metadata.yaml with {len(topics)} topics")
    
    def segment_scenarios(self) -> list[ScenarioWindow]:
        """Segment the bag into individual scenario runs using /social_task."""
        if not self.social_tasks:
            print("Warning: No /social_task messages found. Cannot segment scenarios.")
            return []
        
        scenarios = []
        
        for i, (timestamp, msg) in enumerate(self.social_tasks):
            # Get end time (next task or end of bag)
            if i + 1 < len(self.social_tasks):
                end_time = self.social_tasks[i + 1][0]
            else:
                # Use last message timestamp as end
                all_times = []
                if self.robot_poses:
                    all_times.append(int(self.robot_poses[-1].timestamp * 1e9))
                if self.human_odom:
                    all_times.append(self.human_odom[-1][0])
                end_time = max(all_times) if all_times else timestamp + 60_000_000_000  # +60s fallback
            
            scenarios.append(ScenarioWindow(
                name=msg.data,
                start_time=timestamp,
                end_time=end_time
            ))
        
        print(f"Found {len(scenarios)} scenario runs")
        return scenarios
    
    def _filter_poses_by_window(self, poses: List[TimestampedPose], window: ScenarioWindow) -> List[TimestampedPose]:
        """Filter poses to those within the scenario window."""
        start_s = window.start_time / 1e9
        end_s = window.end_time / 1e9
        return [p for p in poses if start_s <= p.timestamp < end_s]
    
    def _filter_by_window(self, messages: list, window: ScenarioWindow) -> list:
        """Filter messages to those within the scenario window."""
        return [(t, m) for t, m in messages if window.start_time <= t < window.end_time]
    
    def _odom_to_pose(self, timestamp: int, msg) -> TimestampedPose:
        """Convert Odometry message to TimestampedPose with yaw."""
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        
        return TimestampedPose(
            timestamp=timestamp / 1e9,  # Convert ns to seconds
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            yaw=yaw,
            vx=msg.twist.twist.linear.x,
            vy=msg.twist.twist.linear.y
        )
    
    def _sync_poses(self, robot_poses: List[TimestampedPose], 
                    human_poses: List[TimestampedPose]) -> List[Tuple]:
        """Synchronize robot and human poses by nearest timestamp."""
        if not robot_poses or not human_poses:
            return []
        
        synced = []
        human_idx = 0
        
        for robot_pose in robot_poses:
            # Find closest human pose
            while (human_idx < len(human_poses) - 1 and 
                   abs(human_poses[human_idx + 1].timestamp - robot_pose.timestamp) <
                   abs(human_poses[human_idx].timestamp - robot_pose.timestamp)):
                human_idx += 1
            
            # Check if within tolerance
            if abs(human_poses[human_idx].timestamp - robot_pose.timestamp) <= SYNC_TOLERANCE:
                synced.append((robot_pose, human_poses[human_idx]))
        
        return synced
    
    def compute_psc(self, synced_poses: List[Tuple], threshold: float = PSC_THRESHOLD) -> float:
        """Compute Personal Space Compliance using edge-to-edge distance."""
        if not synced_poses:
            return 100.0  # No data = compliant by default
        
        compliant = 0
        for robot, human in synced_poses:
            # Edge-to-edge distance accounting for footprints
            edge_dist = compute_edge_distance(robot.x, robot.y, robot.yaw, human.x, human.y)
            if edge_dist >= threshold:
                compliant += 1
        
        return (compliant / len(synced_poses)) * 100.0
    
    def compute_min_ttc(self, synced_poses: List[Tuple]) -> float:
        """
        Compute minimum Time-to-Collision using edge-to-edge distance.
        
        TTC = edge_distance / closing_speed (only when approaching)
        Collision = edge distance <= 0
        """
        if not synced_poses:
            return float('inf')
        
        min_ttc = float('inf')
        
        for robot, human in synced_poses:
            # Edge-to-edge distance
            edge_dist = compute_edge_distance(robot.x, robot.y, robot.yaw, human.x, human.y)
            
            if edge_dist <= 0:  # Already in collision
                return 0.0
            
            # Relative velocity (robot approaching human means positive closing speed)
            rel_vx = robot.vx - human.vx
            rel_vy = robot.vy - human.vy
            
            # Closing speed approximation (using center-to-center direction)
            rel_x = human.x - robot.x
            rel_y = human.y - robot.y
            center_dist = math.sqrt(rel_x**2 + rel_y**2)
            
            if center_dist < 0.01:
                continue
            
            closing_speed = (rel_vx * rel_x + rel_vy * rel_y) / center_dist
            
            if closing_speed > 0.01:  # Only when approaching
                ttc = edge_dist / closing_speed
                if ttc > 0:
                    min_ttc = min(min_ttc, ttc)
        
        return min_ttc
    
    def compute_min_distance(self, synced_poses: List[Tuple]) -> float:
        """Compute minimum edge-to-edge distance between robot and human."""
        if not synced_poses:
            return float('inf')
        
        min_dist = float('inf')
        for robot, human in synced_poses:
            edge_dist = compute_edge_distance(robot.x, robot.y, robot.yaw, human.x, human.y)
            min_dist = min(min_dist, edge_dist)
        
        return min_dist
    
    def compute_detection_latency(self, window: ScenarioWindow,
                                  human_odom: list, person_markers: list) -> float:
        """
        Compute detection latency.
        
        Time from first /human/odom (human spawn) to first real person detection.
        Real detection = marker with non-zero position in 'detected_persons' namespace.
        """
        if not human_odom:
            return -1.0  # No human data
        
        first_human_time = human_odom[0][0] / 1e9  # Convert to seconds
        
        # Find first marker with actual (non-zero) position
        for timestamp, markers in person_markers:
            for marker in markers.markers:
                # Check for real detection: non-zero position in main namespace
                if marker.ns == 'detected_persons':
                    pos = marker.pose.position
                    if abs(pos.x) > 0.01 or abs(pos.y) > 0.01:
                        detection_time = timestamp / 1e9
                        return detection_time - first_human_time
        
        return -1.0  # No detection
    
    def compute_vlm_latency(self, window: ScenarioWindow,
                           person_markers: list, vlm_responses: list) -> float:
        """
        Compute VLM response latency.
        
        Time from first real detection to first VLM response.
        """
        if not person_markers or not vlm_responses:
            return -1.0
        
        # Find first real detection (non-zero position)
        first_detection_time = None
        for timestamp, markers in person_markers:
            for marker in markers.markers:
                if marker.ns == 'detected_persons':
                    pos = marker.pose.position
                    if abs(pos.x) > 0.01 or abs(pos.y) > 0.01:
                        first_detection_time = timestamp / 1e9
                        break
            if first_detection_time is not None:
                break
        
        if first_detection_time is None:
            return -1.0
        
        # Find first VLM response after detection
        for timestamp, response in vlm_responses:
            response_time = timestamp / 1e9
            if response_time >= first_detection_time:
                return response_time - first_detection_time
        
        return -1.0
    
    def extract_metrics(self) -> list[ScenarioMetrics]:
        """Extract all metrics from the bag file."""
        self.read_bag()
        scenarios = self.segment_scenarios()
        
        if not scenarios:
            print("No scenarios found. Returning empty metrics.")
            return []
        
        # Track run counts for each scenario type
        run_counts = defaultdict(int)
        
        all_metrics = []
        
        for window in scenarios:
            # Parse scenario name from task message (e.g., "frontal_approach_human" -> "frontal_approach")
            scenario_name = window.name.replace('_human', '')
            run_counts[scenario_name] += 1
            run_number = run_counts[scenario_name]
            
            print(f"\nProcessing: {scenario_name} (run {run_number})")
            
            # Filter messages for this window
            robot_poses_window = self._filter_poses_by_window(self.robot_poses, window)
            human_odom_window = self._filter_by_window(self.human_odom, window)
            person_markers_window = self._filter_by_window(self.person_markers, window)
            vlm_responses_window = self._filter_by_window(self.vlm_responses, window)
            
            # Convert human odom to poses
            human_poses = [self._odom_to_pose(t, m) for t, m in human_odom_window]
            
            # Synchronize robot (from TF) and human poses
            synced = self._sync_poses(robot_poses_window, human_poses)
            print(f"  Robot poses: {len(robot_poses_window)}, Human poses: {len(human_poses)}, Synced: {len(synced)}")
            
            # Compute metrics
            psc = self.compute_psc(synced)
            min_ttc = self.compute_min_ttc(synced)
            min_dist = self.compute_min_distance(synced)
            det_lat = self.compute_detection_latency(window, human_odom_window, person_markers_window)
            vlm_lat = self.compute_vlm_latency(window, person_markers_window, vlm_responses_window)
            
            # Duration
            duration = (window.end_time - window.start_time) / 1e9  # Convert ns to seconds
            
            metrics = ScenarioMetrics(
                experiment_name=self.experiment_name,
                scenario_name=scenario_name,
                scenario_run=run_number,
                psc=psc,
                min_ttc=min_ttc,
                min_distance=min_dist,
                detection_latency=det_lat,
                vlm_latency=vlm_lat,
                scenario_duration=duration,
                start_time=window.start_time / 1e9,
                end_time=window.end_time / 1e9
            )
            
            print(f"  PSC: {psc:.1f}%, MinTTC: {min_ttc:.2f}s, MinDist: {min_dist:.2f}m, "
                  f"DetLat: {det_lat:.2f}s, VLMLat: {vlm_lat:.2f}s, Duration: {duration:.1f}s")
            
            all_metrics.append(metrics)
        
        return all_metrics


def find_bags_in_directory(bag_dir: str) -> list[str]:
    """Find all rosbag directories in the given directory."""
    bags = []
    for entry in os.listdir(bag_dir):
        entry_path = os.path.join(bag_dir, entry)
        if os.path.isdir(entry_path):
            # Check if it contains a .db3 file
            for f in os.listdir(entry_path):
                if f.endswith('.db3'):
                    bags.append(entry_path)
                    break
    return sorted(bags)


def main():
    parser = argparse.ArgumentParser(description='Extract metrics from ROS2 bags')
    parser.add_argument('--bag', type=str, help='Path to single bag file/directory')
    parser.add_argument('--bag-dir', type=str, help='Directory containing multiple bags')
    parser.add_argument('--output', type=str, default='metrics_results.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    if not args.bag and not args.bag_dir:
        parser.error("Either --bag or --bag-dir must be specified")
    
    # Collect bag paths
    bag_paths = []
    if args.bag:
        bag_paths.append(args.bag)
    if args.bag_dir:
        bag_paths.extend(find_bags_in_directory(args.bag_dir))
    
    print(f"Found {len(bag_paths)} bags to process")
    
    # Process each bag
    all_metrics = []
    for bag_path in bag_paths:
        try:
            extractor = RosbagMetricsExtractor(bag_path)
            metrics = extractor.extract_metrics()
            all_metrics.extend(metrics)
        except Exception as e:
            print(f"Error processing {bag_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Write results
    if all_metrics:
        write_metrics_csv(all_metrics, args.output)
        print(f"\n{'='*60}")
        print(f"Processed {len(bag_paths)} bags, extracted {len(all_metrics)} scenario metrics")
        print(f"Results saved to: {args.output}")
    else:
        print("No metrics extracted!")


if __name__ == '__main__':
    main()
