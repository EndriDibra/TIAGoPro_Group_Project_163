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
        self.lock = Lock()
        self.callback_group = ReentrantCallbackGroup()
        
        # Load model SDF
        pkg_share = get_package_share_directory('human_spawner')
        model_path = os.path.join(pkg_share, 'models', 'human', 'model.sdf')
        with open(model_path, 'r') as f:
            self.human_sdf = f.read()
        
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
        
        # Load config and spawn initial humans
        config_path = os.path.join(pkg_share, 'config', 'humans.yaml')
        self.load_and_spawn_humans(config_path)
        
        # Control timer (10 Hz)
        self.create_timer(0.1, self.control_loop, callback_group=self.callback_group)
        
        self.get_logger().info('Human controller ready (Velocity Control Mode)')

    def load_and_spawn_humans(self, config_path: str):
        """Load humans from config file and spawn them."""
        if not os.path.exists(config_path):
            self.get_logger().warn(f'Config file not found: {config_path}')
            return
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        humans = config.get('humans', [])
        for human in humans:
            name = human.get('name', f'human_{len(self.spawned_humans)}')
            spawn_pose = human.get('spawn_pose', [0.0, 0.0, 0.0])
            path = human.get('path', [spawn_pose[:2]])
            speed = human.get('speed', 0.5)
            
            self.spawn_human(name, spawn_pose[0], spawn_pose[1], spawn_pose[2], path, speed)

    def spawn_human(self, name: str, x: float, y: float, yaw: float, path: list, speed: float) -> bool:
        """Spawn a human at the given position."""
        if name in self.spawned_humans:
            self.get_logger().warn(f'Human {name} already exists')
            return False
        
        req = SpawnEntity.Request()
        req.name = name
        req.xml = self.human_sdf
        # IMPORTANT: Set robot_namespace so plugin topics are isolated (e.g. /human_1/cmd_vel)
        req.robot_namespace = name
        req.initial_pose = Pose()
        req.initial_pose.position = Point(x=x, y=y, z=0.2)  # Lift slightly to avoid ground collision
        req.initial_pose.orientation = yaw_to_quaternion(yaw)
        req.reference_frame = 'world'
        
        future = self.spawn_client.call_async(req)
        # Verify spawn success
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        
        if future.result() is not None and future.result().success:
            # Create publisher and subscriber for this human
            cmd_vel_topic = f'/{name}/cmd_vel'
            odom_topic = f'/{name}/odom'
            
            pub = self.create_publisher(Twist, cmd_vel_topic, 10)
            
            # We need to create a subscriber that updates the state
            # Using a lambda to capture the name is tricky in loops, but here it's inside a method
            # so `name` is local. However, `self.spawned_humans` modification needs lock.
            
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
            
            # Subscribe to odom to get feedback
            self.create_subscription(
                Odometry,
                odom_topic,
                lambda msg, n=name: self.odom_callback(msg, n),
                10,
                callback_group=self.callback_group
            )
            
            self.get_logger().info(f'Spawned human: {name} at ({x:.2f}, {y:.2f})')
            return True
        else:
            self.get_logger().error(f'Failed to spawn human: {name}')
            return False

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
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.result() is not None and future.result().success:
            with self.lock:
                # Note: We don't destroy publishers/subscribers here explicitly 
                # as standard cleanup is complex, but we remove from the dict so control stops
                del self.spawned_humans[name]
            self.get_logger().info(f'Removed human: {name}')
            return True
        else:
            self.get_logger().error(f'Failed to remove human: {name}')
            return False

    def remove_all_callback(self, request, response):
        """Service callback to remove all humans."""
        names = list(self.spawned_humans.keys())
        success = True
        for name in names:
            if not self.remove_human(name):
                success = False
        
        response.success = success
        response.message = f'Removed {len(names)} humans' if success else 'Some removals failed'
        return response

    def control_loop(self):
        """Control loop for all humans."""
        # Simple throttle for logging
        if int(time.time()) % 5 == 0:
            self.get_logger().info(f'Control loop running, managing {len(self.spawned_humans)} humans', throttle_duration_sec=5.0)

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
        if len(path) < 2:
            return
            
        current_idx = data['current_idx']
        
        # Target waypoint
        # We target the NEXT waypoint (current_idx + 1)
        target_pt = path[(current_idx + 1) % len(path)]
        
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
                    self.spawned_humans[name]['current_idx'] = (current_idx + 1) % len(path)
            # Stop for this step
            msg = Twist()
            data['publisher'].publish(msg)
            return

        # Calculate desired heading
        target_yaw = math.atan2(dy, dx)
        yaw_err = normalize_angle(target_yaw - curr_yaw)
        
        # Control inputs
        # P-controller for turn
        # Speed: reduce if turning sharply
        speed = data['speed']
        
        msg = Twist()
        
        # Simple logic: Turn, then drive? Or curve?
        # Let's use simple differential drive logic
        
        msg.angular.z = 1.0 * yaw_err  # P=1.0 for angular
        
        # Cap angular speed
        msg.angular.z = max(min(msg.angular.z, 1.0), -1.0)
        
        # Linear speed: if facing roughly correct direction, drive
        if abs(yaw_err) < 0.5:  # ~30 degrees
            msg.linear.x = speed
        else:
            msg.linear.x = 0.0  # Turn in place first
            
        # Publish
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
