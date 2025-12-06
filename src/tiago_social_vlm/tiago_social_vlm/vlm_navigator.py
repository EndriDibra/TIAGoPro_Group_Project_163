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
        for m in msg.markers:
            if m.ns == 'detected_persons' and m.action == 0: 
                 count += 1
        
        self.person_count = count
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
            # Placeholder map
            map_img_path = "/tmp/vlm_map_crop.jpg" 
            cv2.imwrite(map_img_path, np.zeros((100,100,3), dtype=np.uint8))
        
        image_path = "/tmp/vlm_current_view.jpg"
        cv2.imwrite(image_path, cv_image)

        prompt = (
            "You are a robot navigation assistant. "
            "I am currently at [0,0] facing forward. "
            "My goal is to reach the target. "
            "There are humans nearby (see image). "
            "Identify the social risk (High/Medium/Low). "
            "Suggest a safe speed multiplier (0.1 to 1.0). "
            "Suggest a short term waypoint relative to me in meters [x, y] to avoid disturbance. "
            "Output JSON."
        )

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
        
        if 'speed' in response:
            speed = float(response['speed'])
            adjusted_speed = speed * self.default_speed
            self._log(f"SPEED_ADJUST: VLM suggested {speed:.2f}, applying {adjusted_speed:.2f} m/s")
            self.set_max_speed(adjusted_speed)
            
        if 'target_pose' in response:
            try:
                # response['target_pose'] is [x, y] relative to base_link
                target = response['target_pose']
                self._log(f"WAYPOINT: VLM suggested target_pose in base_link: [{target[0]}, {target[1]}]")
                
                p = PoseStamped()
                p.header.frame_id = 'base_link'
                # Use Time(0) for TF lookup - means "use latest available transform"
                # Using current time causes "extrapolation into the future" errors
                p.header.stamp = rclpy.time.Time().to_msg()
                p.pose.position.x = float(target[0])
                p.pose.position.y = float(target[1])
                p.pose.orientation.w = 1.0 
                
                # Check if transform is available (use Time() for latest)
                if not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time()):
                    self.get_logger().warn("Cannot transform base_link to map")
                    self._log("TF_ERROR: Cannot transform base_link -> map", level="ERROR")
                    return

                # Transform using latest available data
                p_map = self.tf_buffer.transform(p, 'map')
                self._log(f"TF_TRANSFORM: base_link ({target[0]:.2f}, {target[1]:.2f}) -> map ({p_map.pose.position.x:.3f}, {p_map.pose.position.y:.3f})")
                
                self.get_logger().info(f"Sending VLM Waypoint: {p_map.pose.position.x:.2f}, {p_map.pose.position.y:.2f}")
                self.send_nav_goal(p_map)
                
            except Exception as e:
                self.get_logger().error(f"Failed to process VLM waypoint: {e}")
                self._log(f"WAYPOINT_ERROR: {e}", level="ERROR")

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
