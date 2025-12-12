import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image 
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String

from nav2_msgs.msg import SpeedLimit

import tf2_ros

from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import threading
import time
import logging
import math
from pathlib import Path as FilePath

from tiago_social_vlm.vlm_interface import VLMClient

# --- File Logger Setup ---
def setup_file_logger():
    """Setup a file logger that writes to src/tmp/vlm.log"""
    log_dir = FilePath("src/tmp")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "vlm.log"
    
    logger = logging.getLogger('vlm_file_logger')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler with detailed formatting
    fh = logging.FileHandler(str(log_file), mode='a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger, log_file

class VLMNavigator(Node):
    def __init__(self):
        super().__init__('vlm_navigator')

        # --- File Logger ---
        self.file_logger, self.log_path = setup_file_logger()
        # Clear the log file content before writing new logs
        with open(self.log_path, 'w') as f:
            f.truncate(0)
        self._log("=" * 60)
        self._log("VLM Navigator Started (Supervisor Mode)")
        self._log(f"Log file: {self.log_path}")
        self._log("=" * 60)

        # --- Parameters ---
        self.declare_parameter('controller_server_name', 'controller_server')
        self.declare_parameter('controller_name', 'FollowPath') 
        self.declare_parameter('default_max_speed', 1.0)
        self.declare_parameter('loop_rate', 2.0)  # Fast loop rate (2 Hz = every 0.5 seconds)
        self.declare_parameter('vlm_cooldown', 5.0)  # Seconds between VLM queries
        self.declare_parameter('mistral_api_key', '')
        self.declare_parameter('vlm_backend', 'mock')  # 'mock', 'smol', or 'mistral'
        self.declare_parameter('sim_mode', True) 

        self.controller_server = self.get_parameter('controller_server_name').value
        self.controller_name = self.get_parameter('controller_name').value
        self.default_speed = self.get_parameter('default_max_speed').value
        self.loop_rate = self.get_parameter('loop_rate').value
        self.vlm_cooldown = self.get_parameter('vlm_cooldown').value
        api_key = self.get_parameter('mistral_api_key').value
        if not api_key:
            api_key = os.environ.get('MISTRAL_API_KEY')
        vlm_backend = self.get_parameter('vlm_backend').value

        self._log(f"Parameters: controller={self.controller_server}/{self.controller_name}, default_speed={self.default_speed}, loop_rate={self.loop_rate}Hz, vlm_cooldown={self.vlm_cooldown}s, vlm_backend={vlm_backend}")

        # --- State ---
        self.mode = "IDLE" 
        self.current_path = None # Store the global plan
        self.current_goal = None # Derived from path end
        self.person_count = 0
        self.person_positions = []  # List of (x, y) positions in map frame
        self.latest_person_markers = None
        self.last_person_time = 0.0
        self.human_timeout = 5.0
        self.last_vlm_query_time = 0.0  # Track last VLM query for cooldown
        
        self.latest_rgb = None
        self.latest_map = None
        self.vlm_client = VLMClient(backend=vlm_backend, api_key=api_key)
        self.bridge = CvBridge()
        
        # --- TF Buffer ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- ROS Interfaces ---
        
        # 1. Plan Subscription - Monitor standard Nav2 global plan
        self.plan_sub = self.create_subscription(
            Path,
            '/plan', 
            self.plan_callback,
            10
        )
        self._log("Subscribed to /plan for monitoring Nav2 global path")
        
        # 2. Perception
        self.rgb_sub = self.create_subscription(
            Image, 
            '/head_front_camera/color/image_raw', 
            self.rgb_callback, 
            qos_profile_sensor_data
        )
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.people_sub = self.create_subscription(
            MarkerArray, 
            '/social_costmap/person_markers', 
            self.people_callback, 
            10
        )


        # 3. Speed Control
        self.speed_limit_pub = self.create_publisher(SpeedLimit, 'speed_limit', 10)
        
        # 4. Debug Pubs
        self.debug_img_pub = self.create_publisher(Image, '/vlm/debug_image', 1)
        self.vlm_response_pub = self.create_publisher(String, '/vlm/response', 10)

        # --- Loop Timer ---
        self.timer = self.create_timer(1.0 / self.loop_rate, self.control_loop)
        
        self.get_logger().info("VLM Navigator Initialized. Waiting for goal...")
        self._log("VLM Navigator fully initialized")
        
        self.lock = threading.Lock()

    def _log(self, message, level="INFO"):
        """Write a message to the debug log file."""
        if level == "ERROR":
            self.file_logger.error(message)
        elif level == "WARN":
            self.file_logger.warning(message)
        else:
            self.file_logger.info(message)

    def plan_callback(self, msg):
        """Handle incoming global plan from Nav2."""
        self.get_logger().info(f"Received global plan with {len(msg.poses)} poses")
        
        self.current_path = msg
        # Extract goal from the last pose of the path
        if msg.poses:
            self.current_goal = msg.poses[-1]
            
        self._update_mode()

    def people_callback(self, msg):
        count = 0
        positions = []
        for m in msg.markers:
            if m.ns == 'detected_persons' and m.action == 0:
                count += 1
                # Store position in map frame (markers are typically in map frame)
                positions.append((m.pose.position.x, m.pose.position.y))
        
        self.person_count = count
        self.person_positions = positions
        self.latest_person_markers = msg
        if count > 0:
            self.last_person_time = self.get_clock().now().nanoseconds / 1e9
            
        self._update_mode()

    def rgb_callback(self, msg):
        with self.lock:
            self.latest_rgb = msg

    def map_callback(self, msg):
        with self.lock:
            self.latest_map = msg
    
    def _update_mode(self):
        """Decide mode based on plan existence and person presence."""
        if self.current_path is None:
            self.switch_to_mode("IDLE")
            return

        if self.person_count > 0:
            self.switch_to_mode("VLM_ASSIST")
        else:
            self.switch_to_mode("DIRECT_NAV")

    def switch_to_mode(self, new_mode):
        if self.mode == new_mode:
            return

        self.get_logger().info(f"Switching mode: {self.mode} -> {new_mode}")
        self._log(f"MODE SWITCH: {self.mode} -> {new_mode}")
        self.mode = new_mode
        
        if new_mode == "DIRECT_NAV" or new_mode == "IDLE":
            self._log(f"{new_mode}: Resetting speed to default ({self.default_speed})")
            self.set_max_speed(self.default_speed)
            
        elif new_mode == "VLM_ASSIST":
            self._log("VLM_ASSIST: Immediate VLM query, then cooldown between queries")
            # Reset cooldown to allow immediate query when entering this mode
            self.last_vlm_query_time = 0.0

    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        if self.mode == "DIRECT_NAV":
            # Just ensure speed is max, occasionally re-check
            # The _update_mode called by callbacks handles switching.
            # Timeout check for humans (redundant with people_callback but safe)
            pass

        elif self.mode == "VLM_ASSIST":
            if current_time - self.last_person_time > self.human_timeout:
                self.get_logger().warn(f"Sensor Timeout: No human updates for {self.human_timeout}s while in VLM_ASSIST.")
                self.get_logger().error("Emergency Stop: Sensors may be down. Stopping robot.")
                self._log("SAFETY_STOP: Sensor timeout. Stopping robot.", level="ERROR")
                
                # Force stop
                self.set_max_speed(0.01)  # Use 0.01 instead of 0.0 - Nav2 ignores 0%
                return
            
            # Check cooldown before querying VLM
            time_since_last_query = current_time - self.last_vlm_query_time
            if time_since_last_query >= self.vlm_cooldown:
                self.run_vlm_update()
                self.last_vlm_query_time = current_time

    def run_vlm_update(self):
        if not self.latest_rgb:
            self.get_logger().warn("Waiting for sensor data for VLM...")
            return

        with self.lock:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_rgb, desired_encoding='bgr8')
            map_img = self._generate_map_image()
        
        image_path = "src/tmp/vlm_rgb.jpg"
        map_img_path = "src/tmp/vlm_map.jpg"
        cv2.imwrite(image_path, cv_image)
        cv2.imwrite(map_img_path, map_img)

        # Calculate metrics from path
        heading_deg = None
        distance_to_goal = None
        
        if self.current_path and self.current_path.poses:
            try:
                # 1. Distance to goal (sum of segments)
                # Ensure we transform robot pose to map frame for start point
                if self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time()):
                    trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                    robot_x = trans.transform.translation.x
                    robot_y = trans.transform.translation.y
                    
                    # Find closest point on path to robot (simple approximation: start of path)
                    # Nav2 updates path start to be near robot.
                    
                    dist = 0.0
                    last_x, last_y = robot_x, robot_y
                    
                    # Iterate through path poses
                    for pose in self.current_path.poses:
                        px = pose.pose.position.x
                        py = pose.pose.position.y
                        dist += math.sqrt((px - last_x)**2 + (py - last_y)**2)
                        last_x, last_y = px, py
                    
                    distance_to_goal = dist
                    
                    # 2. Heading to goal (relative to robot)
                    # Use the end of the path as goal
                    goal_x = self.current_path.poses[-1].pose.position.x
                    goal_y = self.current_path.poses[-1].pose.position.y
                    
                    q = trans.transform.rotation
                    robot_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                    
                    angle_to_goal = math.atan2(goal_y - robot_y, goal_x - robot_x)
                    relative_angle = angle_to_goal - robot_yaw
                    while relative_angle > math.pi: relative_angle -= 2 * math.pi
                    while relative_angle < -math.pi: relative_angle += 2 * math.pi
                    heading_deg = math.degrees(relative_angle)
                    
                    self._log(f"METRICS: Dist={distance_to_goal:.1f}m, Heading={heading_deg:.1f}deg")
            except Exception as e:
                self._log(f"METRICS_ERROR: {e}", level="WARN")

        self.get_logger().info("Requesting VLM advice...")
        self._log("VLM_UPDATE: Requesting VLM advice")
        
        # New Signature Call
        response = self.vlm_client.get_navigation_command(image_path, map_img_path, heading_deg, distance_to_goal)
        
        if not response:
            self._log("VLM_UPDATE: No response from VLM", level="WARN")
            return

        self.get_logger().info(f"VLM Response: {response}")
        self._log(f"VLM_RESPONSE: {response}")
        msg = String()
        msg.data = str(response)
        self.vlm_response_pub.publish(msg)
        
        # Apply Speed
        if response.get('speed_valid', False):
            speed = float(response['speed'])
            adjusted_speed = speed * self.default_speed
            action = response.get('action', 'Unknown')
            
            self.get_logger().info(f"VLM Action: {action} ({speed*100:.0f}%)")
            self._log(f"APPLYING: Action={action}, Speed={adjusted_speed:.2f} m/s")
            
            self.set_max_speed(adjusted_speed)
        else:
            self._log(f"SPEED_INVALID: VLM speed failed validation", level="WARN")

    def _generate_map_image(self):
        """Generate a map image with robot, humans, goal, path, and axis for VLM."""
        # Image size and scale (pixels per meter)
        img_size = 400
        scale = 40  # 40 pixels per meter -> 10m x 10m view
        center = img_size // 2
        
        # Create white background
        map_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Get robot position first for OccupancyGrid rendering
        robot_x, robot_y, robot_yaw = 0.0, 0.0, 0.0
        try:
            if self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time()):
                trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                robot_x = trans.transform.translation.x
                robot_y = trans.transform.translation.y
                q = trans.transform.rotation
                robot_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        except Exception as e:
            self._log(f"MAP_GEN: Could not get robot transform: {e}", level="WARN")
        
        # Helper to convert world coords to image coords (robot-centric)
        def world_to_img(wx, wy):
            # Relative to robot position
            rel_x = wx - robot_x
            rel_y = wy - robot_y
            # Rotate by negative robot yaw to align with robot frame
            cos_yaw = math.cos(-robot_yaw)
            sin_yaw = math.sin(-robot_yaw)
            local_x = rel_x * cos_yaw - rel_y * sin_yaw
            local_y = rel_x * sin_yaw + rel_y * cos_yaw
            # Convert to image coords (X forward = up, Y left = left)
            img_x = center - int(local_y * scale)  # Y left -> image left
            img_y = center - int(local_x * scale)  # X forward -> image up
            return img_x, img_y

        # Render OccupancyGrid (walls and obstacles)
        if self.latest_map is not None:
            try:
                occ_data = np.array(self.latest_map.data, dtype=np.int8)
                occ_width = self.latest_map.info.width
                occ_height = self.latest_map.info.height
                resolution = self.latest_map.info.resolution
                origin_x = self.latest_map.info.origin.position.x
                origin_y = self.latest_map.info.origin.position.y
                
                occ_grid = occ_data.reshape((occ_height, occ_width))
                
                # Optimization: Only iterate over relevant grid area? 
                # For now keeping it simple as grid is smallish usually or cpu is fast enough for 400x400
                # Actually, iterating 400x400 pixels is 160k ops, totally fine.
                
                # Create obstacle mask and inflate
                obstacle_mask = (occ_grid > 50).astype(np.uint8)
                inflation_radius_m = 0.3
                inflation_pixels = int(inflation_radius_m / resolution)
                if inflation_pixels > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*inflation_pixels+1, 2*inflation_pixels+1))
                    inflated_obstacles = cv2.dilate(obstacle_mask, kernel)
                else:
                    inflated_obstacles = obstacle_mask

                for img_y in range(img_size):
                    for img_x in range(img_size):
                        local_y = (center - img_x) / scale
                        local_x = (center - img_y) / scale
                        cos_yaw = math.cos(robot_yaw)
                        sin_yaw = math.sin(robot_yaw)
                        world_x = robot_x + local_x * cos_yaw - local_y * sin_yaw
                        world_y = robot_y + local_x * sin_yaw + local_y * cos_yaw
                        grid_x = int((world_x - origin_x) / resolution)
                        grid_y = int((world_y - origin_y) / resolution)
                        
                        if 0 <= grid_x < occ_width and 0 <= grid_y < occ_height:
                            cell_value = occ_grid[grid_y, grid_x]
                            is_inflated = inflated_obstacles[grid_y, grid_x]
                            if cell_value > 50:
                                map_img[img_y, img_x] = [50, 50, 50]
                            elif is_inflated:
                                map_img[img_y, img_x] = [120, 120, 120]
                            elif cell_value < 0:
                                map_img[img_y, img_x] = [200, 200, 200]
            except Exception as e:
                self._log(f"MAP_GEN: Failed to render OccupancyGrid: {e}", level="WARN")
        
        # Draw grid lines
        grid_color = (200, 200, 200)
        for i in range(-5, 6):
            px = center + int(i * scale)
            cv2.line(map_img, (px, 0), (px, img_size), grid_color, 1)
            cv2.line(map_img, (0, px), (img_size, px), grid_color, 1)
        
        # Draw axis labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(map_img, "X (forward)", (center + 10, 20), font, 0.4, (0, 0, 0), 1)
        cv2.putText(map_img, "Y (left)", (10, center - 10), font, 0.4, (0, 0, 0), 1)
        
        # --- NEW: Draw Global Path ---
        if self.current_path and self.current_path.poses:
            path_pts = []
            for pose in self.current_path.poses:
                px = pose.pose.position.x
                py = pose.pose.position.y
                ix, iy = world_to_img(px, py)
                # Filter out crazy outliers if needed, but cv2.polylines handles clips
                path_pts.append([ix, iy])
            
            if len(path_pts) > 1:
                pts = np.array(path_pts, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(map_img, [pts], isClosed=False, color=(255, 0, 0), thickness=2) # Blue path
        
        # Draw robot at center
        cv2.circle(map_img, (center, center), 8, (255, 0, 0), -1)
        arrow_len = 20
        arrow_end = (center, center - arrow_len)
        cv2.arrowedLine(map_img, (center, center), arrow_end, (255, 0, 0), 2, tipLength=0.3)
        
        # Draw humans
        intimate_radius = 0.7
        personal_radius = 1.45
        for (px, py) in self.person_positions:
            img_x, img_y = world_to_img(px, py)
            if 0 <= img_x < img_size and 0 <= img_y < img_size:
                cv2.circle(map_img, (img_x, img_y), 6, (0, 0, 255), -1)
                cv2.circle(map_img, (img_x, img_y), int(intimate_radius * scale), (0, 0, 200), 2)
                cv2.circle(map_img, (img_x, img_y), int(personal_radius * scale), (0, 0, 255), 1)
        
        # Draw goal position (green star)
        if self.current_goal:
            gx = self.current_goal.pose.position.x
            gy = self.current_goal.pose.position.y
            img_x, img_y = world_to_img(gx, gy)
            if 0 <= img_x < img_size and 0 <= img_y < img_size:
                self._draw_star(map_img, (img_x, img_y), 12, (0, 200, 0), -1)
        
        # Draw scale indicator
        cv2.putText(map_img, "1m", (img_size - 60, img_size - 10), font, 0.4, (0, 0, 0), 1)
        cv2.line(map_img, (img_size - 50, img_size - 20), (img_size - 50 + scale, img_size - 20), (0, 0, 0), 2)
        
        return map_img

    def _draw_star(self, img, center, size, color, thickness):
        """Draw a simple star shape at the given center."""
        cx, cy = center
        pts = []
        for i in range(5):
            angle = math.radians(i * 72 - 90)  # Start from top
            outer_x = int(cx + size * math.cos(angle))
            outer_y = int(cy + size * math.sin(angle))
            pts.append([outer_x, outer_y])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], color)

    def set_max_speed(self, speed):
        """Set the max velocity using speed_limit topic."""
        # Publish to speed_limit topic - Nav2's velocity_smoother listens to this
        # SpeedLimit message uses percentage (0-100) not absolute speed
        # We express speed as a percentage of default_speed
        percentage = min(100.0, (speed / self.default_speed) * 100.0)
        
        msg = SpeedLimit()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.percentage = True  # Using percentage mode
        msg.speed_limit = percentage
        
        self.speed_limit_pub.publish(msg)
        self._log(f"SET_SPEED: Published speed limit {percentage:.1f}% (= {speed:.3f} m/s)")
        self.get_logger().info(f"Set speed limit to {percentage:.1f}% ({speed:.2f} m/s)")


def main(args=None):
    rclpy.init(args=args)
    node = VLMNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
