#!/usr/bin/env python3
"""
Extract VLM response samples with context for explanation evaluation.

Usage (inside Docker):
    python3 extract_vlm_samples.py --input /rosbags --output samples.json

This script extracts VLM responses with query-time context (detections, TTC)
for grounding and consistency evaluation.
"""

import argparse
import json
import os
import sys
import math
import ast
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# ROS imports (must be run inside Docker with ROS2)
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    print("Error: Must run inside Docker with ROS2 environment")
    sys.exit(1)


# Average VLM latencies from experiments (seconds)
VLM_LATENCIES = {
    'mistral': 4.3,
    'smol': 5.6,
    'novlm': 0.8,  # Mock VLM is fast
}

# Scenario goals from tiago_social_cmd/config/scenarios.yaml
# Maps task_msg to (goal_x, goal_y)
SCENARIO_GOALS = {
    'frontal_approach_human': (3.0, 3.0),
    'intersection_human': (3.0, 0.0),
    'doorway_human': (1.5, 8.0),
}

# Personal space thresholds (meters) - edge-to-edge distances
PSC_THRESHOLD = 1.25  # Personal space threshold
INTIMATE_SPACE = 0.45  # Intimate zone - collision risk
PERSONAL_SPACE = 1.2   # Personal space

# Footprint definitions (from extract_metrics.py)
HUMAN_RADIUS = 0.25  # meters - human as circle
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


def quaternion_to_yaw(q):
    """Convert quaternion to yaw angle."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class Detection:
    """A single human detection."""
    x: float
    y: float
    distance: float
    angle: float  # Angle in degrees from robot forward (0 = ahead, 90 = left, -90 = right)


@dataclass
class VLMSample:
    """A single VLM response sample with context."""
    sample_id: str
    experiment: str
    response_time: float
    query_time: float
    vlm_response: Dict
    context_at_query: Dict
    labels: Dict


def find_bags_in_directory(bag_dir: str) -> List[str]:
    """Find all rosbag directories in the given directory."""
    bags = []
    for entry in sorted(os.listdir(bag_dir)):
        entry_path = os.path.join(bag_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry.lower() in ['old', 'archive', 'backup', '.old']:
            continue
        for f in os.listdir(entry_path):
            if f.endswith('.db3'):
                bags.append(entry_path)
                break
    return bags


def get_experiment_name(bag_path: str) -> str:
    """Extract experiment name from bag path."""
    basename = os.path.basename(bag_path)
    for exp in ['mistral', 'smol', 'novlm', 'notrack']:
        if exp in basename.lower():
            return exp
    return 'unknown'


def parse_vlm_response(response_str: str) -> Optional[Dict]:
    """Parse VLM response string to dict."""
    try:
        # Response is stored as string representation of dict
        return ast.literal_eval(response_str)
    except:
        return None


def compute_distance(robot_x: float, robot_y: float, 
                    human_x: float, human_y: float) -> float:
    """Compute distance between robot and human."""
    return math.sqrt((robot_x - human_x)**2 + (robot_y - human_y)**2)


def compute_ttc(robot_pose, human_pose, robot_vel, human_vel) -> float:
    """
    Compute time-to-collision using edge-to-edge distance.
    Accounts for robot and human footprints.
    """
    # Relative position
    dx = human_pose[0] - robot_pose[0]
    dy = human_pose[1] - robot_pose[1]
    
    # Relative velocity
    dvx = human_vel[0] - robot_vel[0]
    dvy = human_vel[1] - robot_vel[1]
    
    # Center-to-center distance
    center_distance = math.sqrt(dx*dx + dy*dy)
    
    # Combined footprint radius (robot ~0.35m avg + human 0.25m)
    combined_radius = ROBOT_HALF_LENGTH + HUMAN_RADIUS  # 0.35 + 0.25 = 0.6m
    
    # Edge-to-edge distance
    edge_distance = center_distance - combined_radius
    
    if edge_distance < 0.01:
        return 0.0  # Already colliding
    
    # Dot product of relative position and velocity gives closing speed
    closing_speed = -(dx*dvx + dy*dvy) / center_distance
    
    if closing_speed <= 0:
        return float('inf')  # Not approaching
    
    # Time to edge collision
    return edge_distance / closing_speed


def point_to_line_distance(px: float, py: float, 
                           x1: float, y1: float, 
                           x2: float, y2: float) -> float:
    """
    Compute perpendicular distance from point (px, py) to line segment (x1,y1)-(x2,y2).
    Returns the closest distance to the line segment (not infinite line).
    """
    # Vector from p1 to p2
    dx = x2 - x1
    dy = y2 - y1
    
    # Length squared of the segment
    length_sq = dx*dx + dy*dy
    
    if length_sq < 0.0001:
        # Degenerate case: start and end are same point
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Parameter t for closest point on line
    # Clamp to [0,1] to stay on segment
    t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / length_sq))
    
    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance to closest point
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def classify_path_distance(distance: float) -> str:
    """Classify path distance into space zones."""
    if distance < INTIMATE_SPACE:
        return "intimate"  # Human blocks path - collision risk
    elif distance < PERSONAL_SPACE:
        return "personal"  # Human in personal space of path
    else:
        return "clear"     # Path is clear


class VLMSampleExtractor:
    """Extract VLM samples from rosbags."""
    
    def __init__(self, bag_path: str, bag_id: str = ""):
        self.bag_path = bag_path
        self.bag_id = bag_id
        self.experiment = get_experiment_name(bag_path)
        self.latency = VLM_LATENCIES.get(self.experiment, 5.0)
        
        # Data storage
        self.vlm_responses = []  # (timestamp, response_dict)
        self.person_markers = []  # (timestamp, markers_msg)
        self.human_odom = []  # (timestamp, odom_msg)
        self.robot_odom = []  # (timestamp, odom_msg)
        self.social_tasks = []  # (timestamp, task_name)
    
    def read_bag(self):
        """Read all relevant topics from the bag."""
        print(f"Reading bag: {self.bag_path}")
        
        storage_options = StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        topic_types = {ti.name: ti.type for ti in reader.get_all_topics_and_types()}
        
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            
            if topic == '/vlm/response':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                parsed = parse_vlm_response(msg.data)
                if parsed:
                    self.vlm_responses.append((timestamp, parsed))
            
            elif topic == '/social_costmap/person_markers':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.person_markers.append((timestamp, msg))
            
            elif topic == '/human/odom':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.human_odom.append((timestamp, msg))
            
            elif topic == '/mobile_base_controller/odom':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.robot_odom.append((timestamp, msg))
            
            elif topic == '/social_task':
                msg_type = get_message(topic_types[topic])
                msg = deserialize_message(data, msg_type)
                self.social_tasks.append((timestamp, msg.data))
        
        print(f"  VLM responses: {len(self.vlm_responses)}")
        print(f"  Person markers: {len(self.person_markers)}")
        print(f"  Human odom: {len(self.human_odom)}")
        print(f"  Robot odom: {len(self.robot_odom)}")
    
    def find_nearest_markers(self, query_time_ns: int, robot_pose: Optional[Tuple] = None, 
                              tolerance_s: float = 0.5) -> List[Detection]:
        """Find person markers closest to query time, relative to robot pose."""
        query_time = query_time_ns / 1e9
        best_markers = None
        best_diff = float('inf')
        
        for ts, markers in self.person_markers:
            t = ts / 1e9
            diff = abs(t - query_time)
            if diff < best_diff and diff < tolerance_s:
                best_diff = diff
                best_markers = markers
        
        if best_markers is None:
            return []
        
        # Default robot pose if not provided
        if robot_pose is None:
            robot_x, robot_y, robot_yaw = 0.0, 0.0, 0.0
        else:
            robot_x, robot_y, robot_yaw = robot_pose
        
        # Extract detections from markers
        detections = []
        for m in best_markers.markers:
            if m.ns == 'detected_persons' and m.action == 0:
                # Marker position in map frame
                human_x = m.pose.position.x
                human_y = m.pose.position.y
                
                # Transform to robot-relative coordinates
                dx = human_x - robot_x
                dy = human_y - robot_y
                
                # Rotate by -robot_yaw to get robot frame
                cos_yaw = math.cos(-robot_yaw)
                sin_yaw = math.sin(-robot_yaw)
                rel_x = dx * cos_yaw - dy * sin_yaw
                rel_y = dx * sin_yaw + dy * cos_yaw
                
                # Distance and angle from robot
                dist = math.sqrt(rel_x**2 + rel_y**2)
                # Angle: 0 = forward, positive = left, negative = right
                angle = math.degrees(math.atan2(rel_y, rel_x))
                
                detections.append(Detection(x=round(human_x, 3), y=round(human_y, 3), 
                                           distance=round(dist, 3),
                                           angle=round(angle, 1)))
        
        return detections
    
    def find_nearest_odom(self, odom_list: list, query_time_ns: int) -> Optional[Tuple]:
        """Find odometry closest to query time. Returns ((x, y, yaw), (vx, vy))."""
        query_time = query_time_ns / 1e9
        best_odom = None
        best_diff = float('inf')
        
        for ts, odom in odom_list:
            t = ts / 1e9
            diff = abs(t - query_time)
            if diff < best_diff:
                best_diff = diff
                best_odom = odom
        
        if best_odom is None:
            return None
        
        pos = best_odom.pose.pose.position
        vel = best_odom.twist.twist.linear
        yaw = quaternion_to_yaw(best_odom.pose.pose.orientation)
        return ((pos.x, pos.y, yaw), (vel.x, vel.y))
    
    def find_active_goal(self, query_time_ns: int) -> Optional[Tuple[float, float]]:
        """Find the active navigation goal at query time."""
        query_time = query_time_ns / 1e9
        
        # Find most recent social_task before query time
        active_task = None
        for ts, task_name in self.social_tasks:
            t = ts / 1e9
            if t <= query_time:
                active_task = task_name
        
        if active_task and active_task in SCENARIO_GOALS:
            return SCENARIO_GOALS[active_task]
        return None
    
    def extract_samples(self) -> List[VLMSample]:
        """Extract all VLM samples with query-time context."""
        self.read_bag()
        
        if not self.vlm_responses:
            print("  No VLM responses found")
            return []
        
        # Skip novlm/notrack - they have mock responses
        if self.experiment in ['novlm', 'notrack']:
            print(f"  Skipping {self.experiment} (mock VLM)")
            return []
        
        samples = []
        
        for idx, (response_ts, response) in enumerate(self.vlm_responses):
            response_time = response_ts / 1e9
            query_time_ns = int((response_time - self.latency) * 1e9)
            query_time = query_time_ns / 1e9
            
            # Get robot odom first (needed for detection transforms)
            robot_data = self.find_nearest_odom(self.robot_odom, query_time_ns)
            
            # Get robot pose for detection coordinate transform
            robot_pose = robot_data[0] if robot_data else None  # (x, y, yaw)
            
            # Get detections at query time (relative to robot pose)
            detections = self.find_nearest_markers(query_time_ns, robot_pose=robot_pose)
            
            # Get human position for TTC
            human_data = self.find_nearest_odom(self.human_odom, query_time_ns)
            
            # Get active goal
            goal = self.find_active_goal(query_time_ns)
            
            # Compute TTC and min distance (edge-to-edge)
            ttc = float('inf')
            min_distance = float('inf')
            path_distance = None
            path_zone = None
            
            if robot_data and human_data:
                robot_pos, robot_vel = robot_data  # robot_pos = (x, y, yaw)
                human_pos, human_vel = human_data  # human_pos = (x, y, yaw)
                
                # Edge-to-edge distance (robot rectangle to human circle)
                min_distance = compute_edge_distance(
                    robot_pos[0], robot_pos[1], robot_pos[2],  # robot x, y, yaw
                    human_pos[0], human_pos[1]                  # human x, y
                )
                
                # TTC still uses center-to-center for relative motion calc
                ttc = compute_ttc(
                    (robot_pos[0], robot_pos[1]), 
                    (human_pos[0], human_pos[1]), 
                    robot_vel, human_vel
                )
                
                # Compute path interference: distance from human to robot->goal line
                # Then subtract human radius for edge distance to path
                if goal:
                    center_path_dist = point_to_line_distance(
                        human_pos[0], human_pos[1],
                        robot_pos[0], robot_pos[1],
                        goal[0], goal[1]
                    )
                    # Edge-to-edge: subtract human radius
                    path_distance = center_path_dist - HUMAN_RADIUS
                    path_zone = classify_path_distance(path_distance)
            
            # Create sample object
            # Format: experiment_bagid_index (e.g. mistral_0_0042)
            sid = f"{self.experiment}_{idx:04d}"
            if self.bag_id:
                sid = f"{self.experiment}_{self.bag_id}_{idx:04d}"
            
            sample = VLMSample(
                sample_id=sid,
                experiment=self.experiment,
                response_time=round(response_time, 3),
                query_time=round(query_time, 3),
                vlm_response={
                    'observation': response.get('observation', ''),
                    'prediction': response.get('prediction', ''),
                    'action': response.get('action', 'Unknown')
                },
                context_at_query={
                    'detections': [asdict(d) for d in detections],
                    'ttc': round(ttc, 3) if ttc != float('inf') else None,
                    'min_distance': round(min_distance, 3) if min_distance != float('inf') else None,
                    'path_distance': round(path_distance, 3) if path_distance is not None else None,
                    'path_zone': path_zone  # "intimate", "personal", "social", or "clear"
                },
                labels={
                    'grounded': None,
                    'consistent': None
                }
            )
            samples.append(sample)
        
        print(f"  Extracted {len(samples)} samples")
        return samples


def main():
    parser = argparse.ArgumentParser(
        description='Extract VLM samples for explanation evaluation'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Directory containing rosbag folders'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='vlm_samples.json',
        help='Output JSON file (default: vlm_samples.json)'
    )
    parser.add_argument(
        '--max-per-experiment', '-n',
        type=int,
        default=100,
        help='Maximum samples per experiment (default: 100)'
    )
    parser.add_argument(
        '--preserve-labels', '-p',
        action='store_true',
        help='Preserve labels from existing output file'
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"Error: {args.input} is not a directory")
        sys.exit(1)
    
    bags = find_bags_in_directory(args.input)
    print(f"Found {len(bags)} rosbags")
    
    # Group by experiment
    all_samples = defaultdict(list)
    
    start_time = time.time()
    
    for bag_idx, bag_path in enumerate(bags):
        elapsed = time.time() - start_time
        if bag_idx > 0:
            avg_per_bag = elapsed / bag_idx
            remaining = avg_per_bag * (len(bags) - bag_idx)
            eta_str = f" | ETA: {int(remaining//60)}m {int(remaining%60)}s"
        else:
            eta_str = ""
        
        print(f"\n{'='*60}")
        print(f"[{bag_idx+1}/{len(bags)}] {os.path.basename(bag_path)}{eta_str}")
        print('='*60)
        
        try:
            # Use bag index as unique ID to prevent ID collisions across bags
            extractor = VLMSampleExtractor(bag_path, bag_id=str(bag_idx))
            samples = extractor.extract_samples()
            
            exp = extractor.experiment
            all_samples[exp].extend(samples)
        except Exception as e:
            print(f"Error processing {bag_path}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # Collect all samples (no limit by default)
    final_samples = []
    for exp, samples in all_samples.items():
        final_samples.extend(samples)
    
    # Preserve labels from existing file if requested
    if args.preserve_labels and os.path.isfile(args.output):
        print(f"\nPreserving labels from existing file: {args.output}")
        try:
            with open(args.output, 'r') as f:
                existing_data = json.load(f)
            
            # Build lookup of existing labels by sample_id
            existing_labels = {}
            for s in existing_data.get('samples', []):
                if s.get('labels') and (s['labels'].get('grounded') is not None or 
                                         s['labels'].get('consistent') is not None):
                    existing_labels[s['sample_id']] = s['labels']
            
            # Apply existing labels to new samples
            preserved_count = 0
            for sample in final_samples:
                if sample.sample_id in existing_labels:
                    sample.labels = existing_labels[sample.sample_id]
                    preserved_count += 1
            
            print(f"  Preserved {preserved_count} labels")
        except Exception as e:
            print(f"  Warning: Could not preserve labels: {e}")
    
    # Convert to JSON-serializable format
    output_data = {
        'metadata': {
            'total_samples': len(final_samples),
            'samples_per_experiment': {exp: len([s for s in final_samples if s.experiment == exp]) 
                                       for exp in all_samples.keys()},
            'vlm_latencies': VLM_LATENCIES,
            'extraction_time_seconds': round(total_time, 1)
        },
        'samples': [asdict(s) for s in final_samples]
    }
    
    # Write output
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE in {int(total_time//60)}m {int(total_time%60)}s")
    print(f"Total samples: {len(final_samples)}")
    for exp, count in output_data['metadata']['samples_per_experiment'].items():
        print(f"  {exp}: {count}")
    print(f"Output saved to: {args.output}")
    print('='*60)


if __name__ == '__main__':
    main()
