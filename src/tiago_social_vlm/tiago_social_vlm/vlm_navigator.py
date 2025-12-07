import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Image 
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String

from nav2_msgs.action import NavigateToPose
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter

import tf2_ros
import tf2_geometry_msgs

from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import threading
import time
import logging
import math
from datetime import datetime
from pathlib import Path

from tiago_social_vlm.vlm_interface import VLMClient

# --- File Logger Setup ---
def setup_file_logger():
    """Setup a file logger that writes to src/tmp/vlm_debug.log"""
    log_dir = Path("src/tmp")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "vlm_debug.log"
    
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
        self._log("VLM Navigator Started")
        self._log(f"Log file: {self.log_path}")
        self._log("=" * 60)

        # --- Parameters ---
        self.declare_parameter('controller_server_name', 'controller_server')
        self.declare_parameter('controller_name', 'FollowPath') 
        self.declare_parameter('default_max_speed', 1.0)
        self.declare_parameter('vlm_loop_rate', 0.2) 
        self.declare_parameter('mistral_api_key', '')
        self.declare_parameter('vlm_backend', 'mock')  # 'mock', 'smol', or 'mistral'
        self.declare_parameter('sim_mode', True) 

        self.controller_server = self.get_parameter('controller_server_name').value
        self.controller_name = self.get_parameter('controller_name').value
        self.default_speed = self.get_parameter('default_max_speed').value
        self.loop_rate = self.get_parameter('vlm_loop_rate').value
        api_key = self.get_parameter('mistral_api_key').value
        if not api_key:
            api_key = os.environ.get('MISTRAL_API_KEY')
        vlm_backend = self.get_parameter('vlm_backend').value

        self._log(f"Parameters: controller={self.controller_server}/{self.controller_name}, default_speed={self.default_speed}, loop_rate={self.loop_rate}, vlm_backend={vlm_backend}")

        # --- State ---
        self.mode = "IDLE" 
        self.current_goal = None
        self.person_count = 0
        self.person_positions = []  # List of (x, y) positions in map frame
        self.latest_person_markers = None
        self.last_person_time = 0.0
        self.human_timeout = 5.0 
        
        self.latest_rgb = None
        self.latest_map = None
        self.vlm_client = VLMClient(backend=vlm_backend, api_key=api_key)
        self.bridge = CvBridge()
        
        # --- TF Buffer ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- ROS Interfaces ---
        
        # 1. Goal Subscription - Using dedicated /vlm/goal_pose to avoid conflict with bt_navigator
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/vlm/goal_pose',  # Dedicated topic for VLM-managed navigation
            self.goal_callback,
            10
        )
        self._log("Subscribed to /vlm/goal_pose for VLM-managed goals")
        
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

        # 3. Action Client for Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # 4. Parameter Client for Speed
        self.param_client = self.create_client(SetParameters, f'/{self.controller_server}/set_parameters')
        
        # 5. Debug Pubs
        self.debug_img_pub = self.create_publisher(Image, '/vlm/debug_image', 1)
        self.vlm_response_pub = self.create_publisher(String, '/vlm/response', 10)

        # --- Loop Timer ---
        self.timer = self.create_timer(1.0 / self.loop_rate, self.control_loop)
        
        self.get_logger().info("VLM Navigator Initialized. Waiting for goal...")
        self._log("VLM Navigator fully initialized")
        
        self.lock = threading.Lock()
        self.nav_goal_handle = None

    def _log(self, message, level="INFO"):
        """Write a message to the debug log file."""
        if level == "ERROR":
            self.file_logger.error(message)
        elif level == "WARN":
            self.file_logger.warning(message)
        else:
            self.file_logger.info(message)

    def goal_callback(self, msg):
        """Handle incoming goal requests on /vlm/goal_pose."""
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        self.get_logger().info(f"Received goal on /vlm/goal_pose: ({goal_x:.2f}, {goal_y:.2f})")
        self._log(f"GOAL RECEIVED: x={goal_x:.3f}, y={goal_y:.3f}, frame={msg.header.frame_id}")
        
        self.current_goal = msg
        
        if self.person_count > 0:
            self._log(f"Person count = {self.person_count}, switching to VLM_ASSIST")
            self.switch_to_mode("VLM_ASSIST")
        else:
            self._log("No persons detected, switching to DIRECT_NAV")
            self.switch_to_mode("DIRECT_NAV")

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

    def rgb_callback(self, msg):
        with self.lock:
            self.latest_rgb = msg

    def map_callback(self, msg):
        with self.lock:
            self.latest_map = msg

    def switch_to_mode(self, new_mode):
        if self.mode == new_mode:
            return

        self.get_logger().info(f"Switching mode: {self.mode} -> {new_mode}")
        self._log(f"MODE SWITCH: {self.mode} -> {new_mode}")
        self.mode = new_mode
        
        if new_mode == "DIRECT_NAV":
            self._log(f"DIRECT_NAV: Resetting speed to default ({self.default_speed})")
            self.set_max_speed(self.default_speed)
            if self.current_goal:
                self._log("DIRECT_NAV: Sending stored goal to Nav2")
                self.send_nav_goal(self.current_goal)
            
        elif new_mode == "VLM_ASSIST":
            self._log("VLM_ASSIST: Will query VLM on next control loop")

    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        if self.mode == "DIRECT_NAV":
            if self.person_count > 0:
                self.get_logger().warn("Human detected during Direct Nav! Switching to VLM.")
                # We don't necessarily cancel, just switch mode -> VLM loop will take over
                self.switch_to_mode("VLM_ASSIST")
                return

        elif self.mode == "VLM_ASSIST":
            if current_time - self.last_person_time > self.human_timeout:
                self.get_logger().info("No humans for > 5s. Switching back to Direct Nav.")
                self.switch_to_mode("DIRECT_NAV")
                return
            
            self.run_vlm_update()

    def run_vlm_update(self):
        if not self.latest_rgb:
            self.get_logger().warn("Waiting for sensor data for VLM...")
            return

        with self.lock:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_rgb, desired_encoding='bgr8')
            map_img = self._generate_map_image()
        
        image_path = "src/tmp/vlm_current_view.jpg"
        map_img_path = "src/tmp/vlm_map_crop.jpg"
        cv2.imwrite(image_path, cv_image)
        cv2.imwrite(map_img_path, map_img)

        # Prompt is now centralized in vlm_interface.py
        prompt = ""  # Empty - the backend uses NAVIGATION_PROMPT

        self.get_logger().info("Requesting VLM advice...")
        self._log("VLM_UPDATE: Requesting VLM advice")
        response = self.vlm_client.get_navigation_command(image_path, map_img_path, prompt)
        
        if not response:
            self._log("VLM_UPDATE: No response from VLM", level="WARN")
            return

        self.get_logger().info(f"VLM Response: {response}")
        self._log(f"VLM_RESPONSE: {response}")
        msg = String()
        msg.data = str(response)
        self.vlm_response_pub.publish(msg)
        
        # Check if speed is valid before applying
        if response.get('speed_valid', False):
            speed = float(response['speed'])
            adjusted_speed = speed * self.default_speed
            self._log(f"SPEED_ADJUST: VLM suggested {speed:.2f}, applying {adjusted_speed:.2f} m/s")
            self.set_max_speed(adjusted_speed)
        elif 'speed' in response:
            self._log(f"SPEED_INVALID: VLM speed {response['speed']} failed validation", level="WARN")
            
        # Check if goal is valid before applying
        if response.get('goal_valid', False):
            try:
                target = response['goal']
                self._log(f"WAYPOINT: VLM suggested goal in base_link: [{target[0]}, {target[1]}]")
                
                p = PoseStamped()
                p.header.frame_id = 'base_link'
                p.header.stamp = rclpy.time.Time().to_msg()
                p.pose.position.x = float(target[0])
                p.pose.position.y = float(target[1])
                p.pose.orientation.w = 1.0 
                
                if not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time()):
                    self.get_logger().warn("Cannot transform base_link to map")
                    self._log("TF_ERROR: Cannot transform base_link -> map", level="ERROR")
                    return

                p_map = self.tf_buffer.transform(p, 'map')
                self._log(f"TF_TRANSFORM: base_link ({target[0]:.2f}, {target[1]:.2f}) -> map ({p_map.pose.position.x:.3f}, {p_map.pose.position.y:.3f})")
                
                self.get_logger().info(f"Sending VLM Waypoint: {p_map.pose.position.x:.2f}, {p_map.pose.position.y:.2f}")
                self.send_nav_goal(p_map)
                
            except Exception as e:
                self.get_logger().error(f"Failed to process VLM waypoint: {e}")
                self._log(f"WAYPOINT_ERROR: {e}", level="ERROR")
        else:
            if 'goal' in response:
                self._log(f"GOAL_INVALID: VLM goal {response['goal']} failed validation", level="WARN")

    def _generate_map_image(self):
        """Generate a map image with robot, humans, goal, and axis for VLM."""
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
                
                # Create obstacle mask and inflate it
                obstacle_mask = (occ_grid > 50).astype(np.uint8)
                inflation_radius_m = 0.3  # 30cm robot radius
                inflation_pixels = int(inflation_radius_m / resolution)
                if inflation_pixels > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                       (2*inflation_pixels+1, 2*inflation_pixels+1))
                    inflated_obstacles = cv2.dilate(obstacle_mask, kernel)
                else:
                    inflated_obstacles = obstacle_mask
                
                # For each pixel in our output image, sample the occupancy grid
                for img_y in range(img_size):
                    for img_x in range(img_size):
                        # Convert image coords to robot-local coords
                        local_y = (center - img_x) / scale  # image x -> local y
                        local_x = (center - img_y) / scale  # image y -> local x
                        
                        # Convert robot-local to world coords
                        cos_yaw = math.cos(robot_yaw)
                        sin_yaw = math.sin(robot_yaw)
                        world_x = robot_x + local_x * cos_yaw - local_y * sin_yaw
                        world_y = robot_y + local_x * sin_yaw + local_y * cos_yaw
                        
                        # Convert world to grid coords
                        grid_x = int((world_x - origin_x) / resolution)
                        grid_y = int((world_y - origin_y) / resolution)
                        
                        if 0 <= grid_x < occ_width and 0 <= grid_y < occ_height:
                            cell_value = occ_grid[grid_y, grid_x]
                            is_inflated = inflated_obstacles[grid_y, grid_x]
                            
                            if cell_value > 50:  # Actual wall
                                map_img[img_y, img_x] = [50, 50, 50]  # Dark gray
                            elif is_inflated:  # Inflated zone
                                map_img[img_y, img_x] = [120, 120, 120]  # Medium gray
                            elif cell_value < 0:  # Unknown
                                map_img[img_y, img_x] = [200, 200, 200]  # Light gray
            except Exception as e:
                self._log(f"MAP_GEN: Failed to render OccupancyGrid: {e}", level="WARN")
        
        # Draw grid lines (every 1 meter)
        grid_color = (200, 200, 200)
        for i in range(-5, 6):
            px = center + int(i * scale)
            cv2.line(map_img, (px, 0), (px, img_size), grid_color, 1)
            cv2.line(map_img, (0, px), (img_size, px), grid_color, 1)
        
        # Draw axis labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(map_img, "X (forward)", (center + 10, 20), font, 0.4, (0, 0, 0), 1)
        cv2.putText(map_img, "Y (left)", (10, center - 10), font, 0.4, (0, 0, 0), 1)
        
        # Robot position already obtained above for OccupancyGrid rendering
        
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
        
        # Draw robot at center (blue dot with arrow)
        cv2.circle(map_img, (center, center), 8, (255, 0, 0), -1)  # Blue dot
        arrow_len = 20
        arrow_end = (center, center - arrow_len)  # Arrow pointing up (forward)
        cv2.arrowedLine(map_img, (center, center), arrow_end, (255, 0, 0), 2, tipLength=0.3)
        
        # Draw humans with proxemics circles (red)
        intimate_radius = 0.7  # meters
        personal_radius = 1.45   # meters
        for (px, py) in self.person_positions:
            img_x, img_y = world_to_img(px, py)
            if 0 <= img_x < img_size and 0 <= img_y < img_size:
                # Red dot for person
                cv2.circle(map_img, (img_x, img_y), 6, (0, 0, 255), -1)
                # Intimate space circle (inner, darker red)
                cv2.circle(map_img, (img_x, img_y), int(intimate_radius * scale), (0, 0, 200), 2)
                # Personal space circle (outer, lighter red)
                cv2.circle(map_img, (img_x, img_y), int(personal_radius * scale), (0, 0, 255), 1)
        
        # Draw goal position (green star)
        if self.current_goal:
            gx = self.current_goal.pose.position.x
            gy = self.current_goal.pose.position.y
            img_x, img_y = world_to_img(gx, gy)
            if 0 <= img_x < img_size and 0 <= img_y < img_size:
                # Draw star shape
                self._draw_star(map_img, (img_x, img_y), 12, (0, 200, 0), -1)
        
        # Draw scale indicator
        cv2.putText(map_img, "1m", (img_size - 60, img_size - 10), font, 0.4, (0, 0, 0), 1)
        cv2.line(map_img, (img_size - 50, img_size - 20), (img_size - 50 + scale, img_size - 20), (0, 0, 0), 2)
        
        return map_img
    
    def _draw_star(self, img, center, size, color, thickness):
        """Draw a simple star shape at the given center."""
        cx, cy = center
        # Draw as overlapping triangles for simplicity
        pts = []
        for i in range(5):
            angle = math.radians(i * 72 - 90)  # Start from top
            outer_x = int(cx + size * math.cos(angle))
            outer_y = int(cy + size * math.sin(angle))
            pts.append([outer_x, outer_y])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], color)

    def set_max_speed(self, speed):
        """Set the max velocity parameter on the controller server."""
        self._log(f"SET_SPEED: Attempting to set {self.controller_name}.max_vel_x = {speed:.3f}")
        
        if not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Controller parameter service not available")
            self._log(f"SET_SPEED_ERROR: Service /{self.controller_server}/set_parameters not available", level="ERROR")
            return

        param_name = f"{self.controller_name}.max_vel_x" 
        param = Parameter(param_name, Parameter.Type.DOUBLE, speed)
        
        req = SetParameters.Request()
        req.parameters = [param.to_parameter_msg()]
        
        future = self.param_client.call_async(req)
        self._log(f"SET_SPEED: Sent async request to set {param_name} = {speed:.3f}")
        self.get_logger().info(f"Set max speed to {speed:.2f} m/s")

    def send_nav_goal(self, pose_stamped):
        """Send a navigation goal to Nav2."""
        x = pose_stamped.pose.position.x
        y = pose_stamped.pose.position.y
        frame = pose_stamped.header.frame_id
        
        self._log(f"NAV_GOAL: Sending goal to Nav2: x={x:.3f}, y={y:.3f}, frame={frame}")
        
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("NavigateToPose action server not available!")
            self._log("NAV_GOAL_ERROR: navigate_to_pose action server not available", level="ERROR")
            return
            
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped
        self.nav_client.send_goal_async(goal_msg)
        self._log(f"NAV_GOAL: Goal sent successfully to navigate_to_pose")

def main(args=None):
    rclpy.init(args=args)
    node = VLMNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
