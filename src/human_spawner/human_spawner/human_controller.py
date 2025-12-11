#!/usr/bin/env python3
"""
Human Controller Node - Spawns and moves humans in Gazebo using velocity control.

Services provided:
  - /human_spawner/remove_all: Remove all spawned humans
"""

import math
import os
import time
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory

from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from std_msgs.msg import String
import yaml


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert yaw angle to quaternion."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def quaternion_to_yaw(q: Quaternion) -> float:
    """Extract yaw from quaternion."""
    return 2.0 * math.atan2(q.z, q.w)


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class HumanController(Node):
    """ROS2 node for spawning and moving humans in Gazebo."""

    def __init__(self):
        super().__init__('human_controller')
        
        self.spawned_humans: dict = {}
        self.models: dict = {}
        self.triggers: dict = {}
        self.lock = Lock()
        self.callback_group = ReentrantCallbackGroup()
        
        # Gazebo service clients
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        
        # Wait for spawn/delete services
        self.get_logger().info('Waiting for Gazebo spawn/delete services...')
        self.spawn_client.wait_for_service(timeout_sec=60.0)
        self.delete_client.wait_for_service(timeout_sec=60.0)
        self.get_logger().info('Spawn/delete services available')
        
        # Create service for external control
        self.create_service(Trigger, '/human_spawner/remove_all', self.remove_all_callback)
        
        # Subscriber for social tasks
        self.create_subscription(String, '/social_task', self.task_callback, 10)

        # Load config and spawn initial humans
        pkg_share = get_package_share_directory('human_spawner')
        config_path = os.path.join(pkg_share, 'config', 'humans.yaml')
        self.load_config(config_path)
        
        # Control timer (10 Hz)
        self.create_timer(0.1, self.control_loop, callback_group=self.callback_group)
        
        self.get_logger().info('Human controller ready (Trigger Mode)')

    def load_config(self, config_path: str):
        """Load humans and triggers from config file."""
        if not os.path.exists(config_path):
            self.get_logger().warn(f'Config file not found: {config_path}')
            return
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load models
        pkg_share = get_package_share_directory('human_spawner')
        models_config = config.get('models', {})
        
        
        for model_name, relative_path in models_config.items():
            if not relative_path.startswith('/'):
               full_path = os.path.join(pkg_share, relative_path)
            else:
               full_path = relative_path
            
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    self.models[model_name] = f.read()
            else:
                 self.get_logger().error(f"Model file not found: {full_path}")

        # Ensure we have a default
        if 'default' not in self.models and 'humans' in config:
             # Try to load default from standard location if not specified
             default_model_path = os.path.join(pkg_share, 'models', 'human', 'model.sdf')
             if os.path.exists(default_model_path):
                 with open(default_model_path, 'r') as f:
                     self.models['default'] = f.read()

        # Load Initial Humans
        initial_humans = config.get('initial_humans', []) or config.get('humans', [])

        for human in initial_humans:
            self.process_spawn_action(human)

        # Load Triggers
        self.triggers = config.get('triggers', {})

    def task_callback(self, msg):
        task_name = msg.data
        self.get_logger().info(f"Received task: {task_name}")
        
        if task_name in self.triggers:
            actions = self.triggers[task_name]
            for action_data in actions:
                self.execute_action(action_data)
        else:
            self.get_logger().info(f"No triggers for task: {task_name}")

    def execute_action(self, data):
        action_type = data.get('action', '')
        if action_type == 'spawn':
            self.process_spawn_action(data)
        elif action_type == 'destroy':
            name = data.get('name')
            if name:
                self.remove_human(name)
        elif action_type == 'animate':
            self.process_animate_action(data)

    def process_spawn_action(self, data):
        name = data.get('name', f'human_{len(self.spawned_humans)}')
        model_key = data.get('model', 'default')
        spawn_pose = data.get('spawn_pose', None)
        # Check for 'pose' alias from plan
        if not spawn_pose:
            spawn_pose = data.get('pose', [0.0, 0.0, 0.0])
            
        path = data.get('path', [spawn_pose[:2]])
        speed = data.get('speed', 0.5)

        # Get SDF
        sdf_xml = self.models.get(model_key)
        if not sdf_xml:
            # Fallback to default if exists
            sdf_xml = self.models.get('default')
        
        if sdf_xml:
            self.spawn_human(name, sdf_xml, spawn_pose[0], spawn_pose[1], spawn_pose[2], path, speed)
        else:
            self.get_logger().error(f"No model found for key '{model_key}' (and no default)")

    def process_animate_action(self, data):
        name = data.get('name')
        if name not in self.spawned_humans:
             self.get_logger().warn(f"Cannot animate {name}, not found.")
             return

        animation = data.get('animation')
        if animation:
            self.get_logger().info(f"Triggering animation '{animation}' for {name}")
        
        new_path = data.get('path')
        if new_path:
             with self.lock:
                 self.spawned_humans[name]['path'] = new_path
                 self.spawned_humans[name]['current_idx'] = 0
             self.get_logger().info(f"Updated path for {name}")

    def spawn_human(self, name: str, xml: str, x: float, y: float, yaw: float, path: list, speed: float) -> bool:
        """Spawn a human at the given position."""
        if name in self.spawned_humans:
            self.get_logger().warn(f'Human {name} already exists')
            return False
        
        req = SpawnEntity.Request()
        req.name = name
        req.xml = xml
        # IMPORTANT: Set robot_namespace so plugin topics are isolated (e.g. /human_1/cmd_vel)
        req.robot_namespace = name
        req.initial_pose = Pose()
        req.initial_pose.position = Point(x=x, y=y, z=0.2)
        req.initial_pose.orientation = yaw_to_quaternion(yaw)
        req.reference_frame = 'world'
        
        future = self.spawn_client.call_async(req)
        future.add_done_callback(lambda f: self.spawn_done_callback(f, name, x, y, yaw, path, speed))
        return True

    def spawn_done_callback(self, future, name, x, y, yaw, path, speed):
        try:
            result = future.result()
            if result and result.success:
                cmd_vel_topic = f'/{name}/cmd_vel'
                odom_topic = f'/{name}/odom'
                
                pub = self.create_publisher(Twist, cmd_vel_topic, 10)
                
                with self.lock:
                    self.spawned_humans[name] = {
                        'path': path,
                        'speed': speed,
                        'current_idx': 0,
                        'publisher': pub,
                        'last_odom_x': x,
                        'last_odom_y': y,
                        'last_odom_yaw': yaw,
                        'odom_received': False
                    }
                
                self.create_subscription(
                    Odometry,
                    odom_topic,
                    lambda msg, n=name: self.odom_callback(msg, n),
                    10,
                    callback_group=self.callback_group
                )
                self.get_logger().info(f'Spawned human: {name} at ({x:.2f}, {y:.2f})')
            else:
                self.get_logger().error(f'Failed to spawn human: {name}. Gazebo: {result.status_message if result else "Unknown"}')
        except Exception as e:
            self.get_logger().error(f'Exception spawning {name}: {e}')

    def odom_callback(self, msg: Odometry, name: str):
        """Update human state from odom."""
        with self.lock:
            if name in self.spawned_humans:
                self.spawned_humans[name]['last_odom_x'] = msg.pose.pose.position.x
                self.spawned_humans[name]['last_odom_y'] = msg.pose.pose.position.y
                self.spawned_humans[name]['last_odom_yaw'] = quaternion_to_yaw(msg.pose.pose.orientation)
                self.spawned_humans[name]['odom_received'] = True

    def remove_human(self, name: str) -> bool:
        """Remove a human by name."""
        if name not in self.spawned_humans:
            self.get_logger().warn(f'Human {name} not found')
            return False
        
        req = DeleteEntity.Request()
        req.name = name
        
        future = self.delete_client.call_async(req)
        future.add_done_callback(lambda f: self.remove_done_callback(f, name))
        return True

    def remove_done_callback(self, future, name):
        try:
            if future.result() and future.result().success:
                with self.lock:
                    if name in self.spawned_humans:
                        del self.spawned_humans[name]
                self.get_logger().info(f'Removed human: {name}')
            else:
                 self.get_logger().error(f'Failed to remove human: {name}')
        except Exception:
            pass

    def remove_all_callback(self, request, response):
        """Service callback to remove all humans."""
        names = list(self.spawned_humans.keys())
        for name in names:
            self.remove_human(name)
        
        response.success = True
        response.message = f'Triggered removal of {len(names)} humans'
        return response

    def control_loop(self):
        """Control loop for all humans."""

        with self.lock:
            humans_copy = list(self.spawned_humans.items())
        
        for name, data in humans_copy:
            try:
                self.control_human(name, data)
            except Exception as e:
                self.get_logger().error(f'Error controlling {name}: {e}')

    def control_human(self, name, data):
        """Calculate and publish control for a single human."""
        path = data['path']
        if not path or len(path) < 1:
            return
            
        current_idx = data['current_idx']
        
        # Check if we've reached the end of the path
        if current_idx >= len(path) - 1:
            # Already at end of path - stop the human
            msg = Twist()
            data['publisher'].publish(msg)
            return
        
        # Target waypoint is next point in path
        target_pt = path[current_idx + 1]
        
        # Current position (from odom or initial spawn)
        curr_x = data['last_odom_x']
        curr_y = data['last_odom_y']
        curr_yaw = data['last_odom_yaw']
        
        # Helper: Get distance to target
        dx = target_pt[0] - curr_x
        dy = target_pt[1] - curr_y
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Check if reached waypoint (tolerance 0.3m)
        if dist < 0.3:
            # Advance to next waypoint
            with self.lock:
                if name in self.spawned_humans:
                    self.spawned_humans[name]['current_idx'] = current_idx + 1
            
            # Stop for this step
            msg = Twist()
            data['publisher'].publish(msg)
            return

        # Calculate desired heading
        target_yaw = math.atan2(dy, dx)
        
        yaw_err = normalize_angle(target_yaw - curr_yaw)
        
        # Control inputs
        # P-controller for turn
        speed = data['speed']
        
        msg = Twist()
        msg.angular.z = 1.0 * yaw_err 
        msg.angular.z = max(min(msg.angular.z, 1.0), -1.0)
        
        if abs(yaw_err) < 0.5: 
            msg.linear.x = speed
        else:
            msg.linear.x = 0.0 
            
        data['publisher'].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HumanController()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
