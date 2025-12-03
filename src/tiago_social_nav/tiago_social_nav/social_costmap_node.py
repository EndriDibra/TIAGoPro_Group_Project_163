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
        self.declare_parameter('rgb_camera_info_topic', '/head_front_camera/color/camera_info')
        
        # Get parameters
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.rgb_camera_info_topic = self.get_parameter('rgb_camera_info_topic').get_parameter_value().string_value
        
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
        
        # Camera Info Subscribers
        self.depth_camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.depth_camera_info_callback,
            qos_profile_sensor_data)
            
        self.rgb_camera_info_sub = self.create_subscription(
            CameraInfo,
            self.rgb_camera_info_topic,
            self.rgb_camera_info_callback,
            qos_profile_sensor_data)
            
        self.depth_camera_model = None
        self.rgb_camera_model = None
        
        self.bridge = CvBridge()
        
        self.get_logger().info('Social Costmap Node started (RGB-D mode)')

    def depth_camera_info_callback(self, msg):
        if self.depth_camera_model is None:
            self.depth_camera_model = msg
            self.get_logger().info(f'Received Depth Camera Info: {msg.width}x{msg.height}, K={msg.k}')

    def rgb_camera_info_callback(self, msg):
        if self.rgb_camera_model is None:
            self.rgb_camera_model = msg
            self.get_logger().info(f'Received RGB Camera Info: {msg.width}x{msg.height}, K={msg.k}')

    def rgbd_callback(self, rgb_msg, depth_msg):
        if self.depth_camera_model is None or self.rgb_camera_model is None:
            self.get_logger().warn('Waiting for camera info...')
            return

        try:
            # Convert images
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            if cv_depth.dtype == np.uint16:
                cv_depth = cv_depth.astype(np.float32) / 1000.0
            
            # Get transforms
            try:
                # Transform from Depth Frame to RGB Frame
                # We need the transform at the time of the image
                transform = self.tf_buffer.lookup_transform(
                    self.rgb_camera_model.header.frame_id,
                    self.depth_camera_model.header.frame_id,
                    rclpy.time.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f'TF Error: {e}')
                return

            # Camera Intrinsics
            depth_K = np.array(self.depth_camera_model.k).reshape(3, 3)
            rgb_K = np.array(self.rgb_camera_model.k).reshape(3, 3)
            
            # Rotation and Translation from Depth to RGB
            q = transform.transform.rotation
            t = transform.transform.translation
            
            # Manual Quaternion to Rotation Matrix
            # q = [x, y, z, w]
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            
            R_depth_to_rgb = np.array([
                [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
            ])
            
            T_depth_to_rgb = np.array([t.x, t.y, t.z])
            
            # We want to map the center pixel of the RGB image to a depth value
            # Strategy: Project all depth points to RGB frame and find the one closest to center?
            # Or: Raycast from RGB center?
            # Better: Create a registered depth image (sparse or dense)
            
            # For efficiency in Python, let's just map the specific point we care about if possible.
            # But we want to save the "depth image" aligned to RGB.
            # So we need to warp the depth image.
            
            # Create a map from RGB pixels to Depth pixels
            # This is expensive in Python. 
            # Alternative: Just pick the center of RGB, project to 3D ray, transform to Depth frame, project to Depth pixel.
            
            rgb_height, rgb_width = cv_rgb.shape[:2]
            u_rgb, v_rgb = rgb_width // 2, rgb_height // 2
            
            # Ray in RGB frame
            # Z = 1
            # X = (u - cx) * Z / fx
            # Y = (v - cy) * Z / fy
            rgb_fx = rgb_K[0, 0]
            rgb_fy = rgb_K[1, 1]
            rgb_cx = rgb_K[0, 2]
            rgb_cy = rgb_K[1, 2]
            
            ray_x = (u_rgb - rgb_cx) / rgb_fx
            ray_y = (v_rgb - rgb_cy) / rgb_fy
            ray_z = 1.0
            
            ray_rgb = np.array([ray_x, ray_y, ray_z])
            
            # Transform ray to Depth frame
            # P_rgb = R * P_depth + T
            # P_depth = R_inv * (P_rgb - T)
            
            # This is tricky because we don't know the depth Z yet.
            # We are looking for a point P_depth such that P_depth projects to (u_depth, v_depth) 
            # and P_rgb projects to (u_rgb, v_rgb).
            
            # Let's try a simpler approach: 
            # 1. We want the depth at (u_rgb, v_rgb).
            # 2. This corresponds to a line in 3D space.
            # 3. We can sample points along this line in the Depth frame and check the depth map? No.
            
            # Correct approach for "Registered Depth":
            # For every pixel in Depth image:
            #   1. Deproject to 3D point in Depth Frame
            #   2. Transform to RGB Frame
            #   3. Project to RGB Image Plane
            #   4. Store depth value at that RGB pixel
            
            # Vectorized implementation
            depth_height, depth_width = cv_depth.shape
            u_grid, v_grid = np.meshgrid(np.arange(depth_width), np.arange(depth_height))
            
            depth_fx = depth_K[0, 0]
            depth_fy = depth_K[1, 1]
            depth_cx = depth_K[0, 2]
            depth_cy = depth_K[1, 2]
            
            Z_depth = cv_depth
            X_depth = (u_grid - depth_cx) * Z_depth / depth_fx
            Y_depth = (v_grid - depth_cy) * Z_depth / depth_fy
            
            # Flatten
            valid_mask = (Z_depth > 0) & (~np.isnan(Z_depth))
            points_depth = np.vstack((X_depth[valid_mask], Y_depth[valid_mask], Z_depth[valid_mask])).T
            
            # Transform to RGB Frame
            # P_rgb = R * P_depth + T
            points_rgb_3d = np.dot(points_depth, R_depth_to_rgb.T) + T_depth_to_rgb
            
            # Project to RGB Image
            X_rgb = points_rgb_3d[:, 0]
            Y_rgb = points_rgb_3d[:, 1]
            Z_rgb = points_rgb_3d[:, 2]
            
            u_proj = (X_rgb * rgb_fx / Z_rgb) + rgb_cx
            v_proj = (Y_rgb * rgb_fy / Z_rgb) + rgb_cy
            
            # Filter valid projections
            in_bounds = (u_proj >= 0) & (u_proj < rgb_width) & (v_proj >= 0) & (v_proj < rgb_height) & (Z_rgb > 0)
            
            u_proj = u_proj[in_bounds].astype(int)
            v_proj = v_proj[in_bounds].astype(int)
            Z_proj = Z_rgb[in_bounds]
            
            # Create registered depth map
            registered_depth = np.zeros((rgb_height, rgb_width), dtype=np.float32)
            registered_depth[v_proj, u_proj] = Z_proj
            
            # Now we have a depth map that matches the RGB image!
            z = registered_depth[v_rgb, u_rgb]
            
            # Save Data
            save_dir = os.path.expanduser('~/src/tmp')
            
            # Save RGB
            cv2.imwrite(os.path.join(save_dir, 'rgb.jpg'), cv_rgb)
            
            # Save Depth (Normalize to 0-255 for visualization)
            depth_vis = cv2.normalize(registered_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, 'depth.png'), depth_vis)
            
            # Save Point
            # We need to calculate the 3D point in Map frame again using the RGB frame info
            x = (u_rgb - rgb_cx) * z / rgb_fx
            y = (v_rgb - rgb_cy) * z / rgb_fy
            
            point_cam = PointStamped()
            point_cam.header = rgb_msg.header
            point_cam.point.x = float(x)
            point_cam.point.y = float(y)
            point_cam.point.z = float(z)
            
            try:
                transform_map = self.tf_buffer.lookup_transform('map', point_cam.header.frame_id, rclpy.time.Time())
                point_map = tf2_geometry_msgs.do_transform_point(point_cam, transform_map)
                
                point_str = f'Time: {rgb_msg.header.stamp.sec}.{rgb_msg.header.stamp.nanosec}\n'
                point_str += f'Pixel: ({u_rgb}, {v_rgb})\n'
                point_str += f'Camera Frame: ({x:.3f}, {y:.3f}, {z:.3f})\n'
                point_str += f'Map Frame: ({point_map.point.x:.3f}, {point_map.point.y:.3f}, {point_map.point.z:.3f})\n'
                
                with open(os.path.join(save_dir, 'point.txt'), 'w') as f:
                    f.write(point_str)
                
                self.get_logger().info(f'Saved registered data. Map Point: ({point_map.point.x:.3f}, {point_map.point.y:.3f}, {point_map.point.z:.3f})')
                
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
