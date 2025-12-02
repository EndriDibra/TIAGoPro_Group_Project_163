import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import message_filters
from rclpy.qos import qos_profile_sensor_data
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

class SocialCostmapNode(Node):
    def __init__(self):
        super().__init__('social_costmap_node')
        
        # Declare parameters
        self.declare_parameter('camera_topic', '/head_front_camera/color/image_raw')
        self.declare_parameter('depth_topic', '/head_front_camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/head_front_camera/depth/camera_info')
        
        # Get parameters
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        
        self.get_logger().info(f'Subscribing to RGB: {self.camera_topic}, Depth: {self.depth_topic}')
        
        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscribers
        self.rgb_sub = message_filters.Subscriber(self, Image, self.camera_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        
        # Synchronizer
        # ApproximateTimeSynchronizer is robust to slight timestamp differences
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.rgbd_callback)
        
        # Camera Info Subscriber (we need intrinsics)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data)
        self.camera_model = None
        
        self.bridge = CvBridge()
        
        self.get_logger().info('Social Costmap Node started (RGB-D mode)')

    def camera_info_callback(self, msg):
        if self.camera_model is None:
            self.camera_model = msg
            self.get_logger().info('Received Camera Info')

    def rgbd_callback(self, rgb_msg, depth_msg):
        if self.camera_model is None:
            self.get_logger().warn('Waiting for camera info...')
            return

        try:
            # Convert images
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            # Depth is usually 16UC1 (mm) or 32FC1 (m). 
            # If 16UC1, convert to float meters.
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            if cv_depth.dtype == np.uint16:
                cv_depth = cv_depth.astype(np.float32) / 1000.0
            
            # Sample center pixel
            height, width = cv_depth.shape
            u, v = width // 2, height // 2
            z = cv_depth[v, u]
            
            if np.isnan(z) or z <= 0:
                self.get_logger().warn(f'Invalid depth at center ({u}, {v}): {z}')
                return
            
            # Deproject to 3D point in camera frame
            # P = [X, Y, Z]
            # u = fx * X / Z + cx
            # v = fy * Y / Z + cy
            # => X = (u - cx) * Z / fx
            # => Y = (v - cy) * Z / fy
            
            K = self.camera_model.k
            fx = K[0]
            cx = K[2]
            fy = K[4]
            cy = K[5]
            
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            # Create PointStamped
            point_cam = PointStamped()
            point_cam.header = depth_msg.header # Use depth header for frame and time
            point_cam.point.x = float(x)
            point_cam.point.y = float(y)
            point_cam.point.z = float(z)
            
            # Transform to map frame
            try:
                transform = self.tf_buffer.lookup_transform('map', point_cam.header.frame_id, rclpy.time.Time())
                point_map = tf2_geometry_msgs.do_transform_point(point_cam, transform)
                
                # Save Data
                save_dir = os.path.expanduser('~/src/tmp')
                
                # Save RGB
                cv2.imwrite(os.path.join(save_dir, 'rgb.jpg'), cv_rgb)
                
                # Save Depth (Normalize to 0-255 for visualization)
                depth_vis = cv2.normalize(cv_depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = depth_vis.astype(np.uint8)
                cv2.imwrite(os.path.join(save_dir, 'depth.png'), depth_vis)
                
                # Save Point
                point_str = f'Time: {depth_msg.header.stamp.sec}.{depth_msg.header.stamp.nanosec}\n'
                point_str += f'Pixel: ({u}, {v})\n'
                point_str += f'Camera Frame: ({x:.3f}, {y:.3f}, {z:.3f})\n'
                point_str += f'Map Frame: ({point_map.point.x:.3f}, {point_map.point.y:.3f}, {point_map.point.z:.3f})\n'
                
                with open(os.path.join(save_dir, 'point.txt'), 'w') as f:
                    f.write(point_str)
                
                self.get_logger().info(f'Saved data. Map Point: ({point_map.point.x:.3f}, {point_map.point.y:.3f}, {point_map.point.z:.3f})')
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f'TF Error: {e}')
                
        except Exception as e:
            self.get_logger().error(f'Error processing RGB-D: {e}')

def main(args=None):
    rclpy.init(args=args)
    social_costmap_node = SocialCostmapNode()
    rclpy.spin(social_costmap_node)
    social_costmap_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
