import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, LaserScan
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from typing import List, Dict, Optional
import message_filters
from rclpy.qos import qos_profile_sensor_data
import tf2_ros
import tf2_geometry_msgs
import threading

# Import our custom modules
from .person_detector import PersonDetector
from .person_localizer import PersonLocalizer
from .social_costmap_publisher import SocialCostmapPublisher
from .tracking import PersonTracker
import time
import math

class SocialCostmapNode(Node):
    def __init__(self):
        super().__init__('social_costmap_node')
        
        # Declare camera parameters
        self.declare_parameter('camera_topic', '/head_front_camera/color/image_raw')
        self.declare_parameter('depth_topic', '/head_front_camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/head_front_camera/depth/camera_info')
        self.declare_parameter('rgb_camera_info_topic', '/head_front_camera/color/camera_info')
        self.declare_parameter('scan_topic', '/scan') # Added scan topic parameter
        
        # Declare YOLO detection parameters
        self.declare_parameter('detection_rate', 5.0)  # Hz
        self.declare_parameter('yolo_model', 'yolo11n-seg.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cuda')  # 'cuda' or 'cpu'
        
        # Declare localization parameters
        self.declare_parameter('localization_method', '3d_centroid')  # or 'min_z_column'
        
        # Declare debugging parameters
        self.declare_parameter('save_debug_images', True)
        self.declare_parameter('debug_dir', '~/src/tmp')
        
        # Get parameters
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.rgb_camera_info_topic = self.get_parameter('rgb_camera_info_topic').get_parameter_value().string_value
        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        
        detection_rate = self.get_parameter('detection_rate').get_parameter_value().double_value
        yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
        confidence = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        device = self.get_parameter('device').get_parameter_value().string_value
        
        localization_method = self.get_parameter('localization_method').get_parameter_value().string_value
        
        self.save_debug = self.get_parameter('save_debug_images').get_parameter_value().bool_value
        self.debug_dir = os.path.expanduser(self.get_parameter('debug_dir').get_parameter_value().string_value)
        
        self.get_logger().info(f'Subscribing to RGB: {self.camera_topic}, Depth: {self.depth_topic}, Scan: {self.scan_topic}')
        self.get_logger().info(f'Detection rate: {detection_rate} Hz, Model: {yolo_model}, Device: {device}')
        self.get_logger().info(f'Localization method: {localization_method}')
        
        # Rate limiting for detection
        self.detection_interval = 1.0 / detection_rate
        self.last_detection_time = 0.0
        
        # Initialize modules
        self.get_logger().info('Initializing YOLO detector...')
        self.detector = PersonDetector(
            model_name=yolo_model,
            confidence_threshold=confidence,
            device=device
        )
        
        self.localizer = PersonLocalizer(method=localization_method)
        self.costmap_publisher_module = SocialCostmapPublisher()
        self.tracker = PersonTracker(decay_time=20.0, distance_threshold=0.8) # Initialize Tracker
        
        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Threading lock for race condition prevention
        self._tracking_lock = threading.Lock()
        self._pending_yolo_detections = None
        
        # ROS Publishers
        # Replaced OccupancyGrid with PointCloud2
        self.social_pub = self.create_publisher(PointCloud2, '/social_obstacles', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/social_costmap/person_markers', 10)
        
        # Subscribers
        self.rgb_sub = message_filters.Subscriber(self, Image, self.camera_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        
        # Laser Subscriber
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile=qos_profile_sensor_data 
        )
        self.last_scan_msg = None
        
        # Unified tracking timer at ~20Hz to avoid race conditions
        self.tracking_timer = self.create_timer(0.05, self._tracking_timer_callback)
        
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
        
        self.get_logger().info('Social Costmap Node (PointCloud2) with YOLO + Laser Tracking ready!')

    def depth_camera_info_callback(self, msg):
        if self.depth_camera_model is None:
            self.depth_camera_model = msg
            self.get_logger().info(f'Received Depth Camera Info: {msg.width}x{msg.height}')

    def rgb_camera_info_callback(self, msg):
        if self.rgb_camera_model is None:
            self.rgb_camera_model = msg
            self.get_logger().info(f'Received RGB Camera Info: {msg.width}x{msg.height}')

    def scan_callback(self, msg):
        """Store latest scan for use in tracking timer."""
        self.last_scan_msg = msg
        # Don't process here - let the unified timer handle tracking
    
    def _tracking_timer_callback(self):
        """Unified tracking callback to avoid race conditions."""
        with self._tracking_lock:
            yolo_detections = self._pending_yolo_detections
            self._pending_yolo_detections = None
        
        # Process tracking with whatever data we have
        self.process_tracking(rgb_detections=yolo_detections, scan_msg=self.last_scan_msg)

    def rgbd_callback(self, rgb_msg, depth_msg):
        """Process synchronized RGB-D data with rate-limited YOLO detection."""
        # Rate limiting for detection
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return  # Skip this frame
        
        self.last_detection_time = current_time
        
        if self.depth_camera_model is None or self.rgb_camera_model is None:
            self.get_logger().warn('Waiting for camera info...')
            return

        try:
            # Convert images
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            if cv_depth.dtype == np.uint16:
                cv_depth = cv_depth.astype(np.float32) / 1000.0
            
            # ========== Step 1: Run YOLO Detection ==========
            # Convert BGR to RGB for YOLO
            cv_rgb_yolo = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect(cv_rgb_yolo)
            
            # ========== Step 2: Create Registered Depth Image ==========
            registered_depth = self._create_registered_depth(
                cv_depth, cv_rgb, rgb_msg.header.frame_id
            )
            
            localized_persons_3d = []
            
            if registered_depth is not None and len(detections) > 0:
                # ========== Step 3: Localize Persons in 3D ==========
                depth_K = np.array(self.rgb_camera_model.k).reshape(3, 3)
                
                # Get transform from camera to map
                try:
                    transform_cam_to_map = self.tf_buffer.lookup_transform(
                        'map',
                        rgb_msg.header.frame_id,
                        rclpy.time.Time()
                    )
                    
                    # Convert to 4x4 matrix
                    q = transform_cam_to_map.transform.rotation
                    t = transform_cam_to_map.transform.translation
                    
                    # Quaternion to rotation matrix
                    qx, qy, qz, qw = q.x, q.y, q.z, q.w
                    R = np.array([
                        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
                        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
                    ])
                    T = np.array([t.x, t.y, t.z])
                    
                    transform_matrix = PersonLocalizer.create_transform_matrix(T, R)
                    
                    # Localize all persons
                    localized_dicts = self.localizer.localize_persons(
                        detections=detections,
                        registered_depth=registered_depth,
                        depth_K=depth_K,
                        depth_to_map_transform=transform_matrix
                    )
                    
                    # Extract just the 3D positions for the tracker
                    for p in localized_dicts:
                        if 'position_3d_map' in p:
                            localized_persons_3d.append(p['position_3d_map'])
                            
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    self.get_logger().warn(f'TF Error (camera to map): {e}')
            
            # ========== Step 4: Queue YOLO detections for tracking timer ==========
            # Don't process tracking here to avoid race condition with scan_callback
            with self._tracking_lock:
                self._pending_yolo_detections = localized_persons_3d

            # ========== Step 6: Save Debug Images (Optional) ==========
            if self.save_debug and len(detections) > 0 and registered_depth is not None:
                # We need to construct dummy localized_persons dicts for debug saving roughly
                # This is a bit disjointed now that tracking is separate, but for debug viz:
                debug_persons = []
                for i, pos in enumerate(localized_persons_3d):
                     debug_persons.append({
                         'confidence': detections[i]['confidence'],
                         'position_3d_map': pos
                     })
                
                self._save_debug_data(cv_rgb, registered_depth, detections, debug_persons)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB-D: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def process_tracking(self, rgb_detections: Optional[List[np.ndarray]], scan_msg: Optional[LaserScan]):
        """
        Main logic to fuse sensors and update tracker.
        rgb_detections: List of [x,y,z] arrays in map frame (from YOLO)
        scan_msg: LaserScan message (to be clustered)
        """
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        laser_clusters_map = []
        
        # Process Laser if available
        if scan_msg is not None:
             laser_clusters_map = self._process_laser_scan(scan_msg)
        
        # Run Tracker Update
        # Note: rgb_detections might be None if this is a laser-only update
        # If rgb_detections is [], it means we ran YOLO and found NOTHING.
        # If rgb_detections is None, it means we didn't run YOLO this frame.
        
        yolo_input = rgb_detections if rgb_detections is not None else []
        
        tracked_persons = self.tracker.update_tracks(
            yolo_detections=yolo_input,
            laser_clusters=laser_clusters_map,
            current_time=current_time
        )
        
        # Publish
        self._publish_tracking_results(tracked_persons, current_time)

    def _process_laser_scan(self, scan_msg: LaserScan) -> List[np.ndarray]:
        """Convert LaserScan to Clusters in Map Frame"""
        clusters = []
        
        ranges = np.array(scan_msg.ranges)
        
        # Robustly generate angles based on actual number of range readings
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        
        # Filter invalid
        valid = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max) & (~np.isinf(ranges)) & (~np.isnan(ranges))
        
        if not np.any(valid):
            return []
            
        r_valid = ranges[valid]
        a_valid = angles[valid]
        
        x = r_valid * np.cos(a_valid)
        y = r_valid * np.sin(a_valid)
        points = np.vstack((x, y)).T
        
        # 2. Cluster points (Euclidean/DBSCAN simplified)
        # Simple jump distance clustering
        if len(points) == 0:
            return []
            
        clusters_list = []
        current_cluster = [points[0]]
        
        # Tuning for leg detection / people
        JUMP_THRESHOLD = 0.5 # meters
        
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i-1])
            if dist < JUMP_THRESHOLD:
                current_cluster.append(points[i])
            else:
                clusters_list.append(np.array(current_cluster))
                current_cluster = [points[i]]
        clusters_list.append(np.array(current_cluster))
        
        # 3. Filter clusters (assume person width < 1.0m and > 0.1m)
        valid_centroids = []
        for c in clusters_list:
            if len(c) < 3: continue # Too few points
            
            width = np.linalg.norm(c[0] - c[-1])
            if 0.05 < width < 1.0:
                centroid = np.mean(c, axis=0)
                valid_centroids.append(centroid) # [x, y] in laser frame
        
        if not valid_centroids:
            return []
            
        # 4. Transform to Map Frame
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                scan_msg.header.frame_id,
                rclpy.time.Time()
            )
            
            # 2D Transform
            q = transform.transform.rotation
            # Yaw from quaternion
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            
            map_centroids = []
            
            cos_theta = np.cos(yaw)
            sin_theta = np.sin(yaw)
            
            for c in valid_centroids:
                # Rotate and translate
                mx = c[0] * cos_theta - c[1] * sin_theta + tx
                my = c[0] * sin_theta + c[1] * cos_theta + ty
                # Add default z=0
                map_centroids.append(np.array([mx, my, 0.0]))
                
            return map_centroids
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            # self.get_logger().warn(f'TF Error (laser to map): {e}')
            return []

    def _publish_tracking_results(self, tracks, timestamp_sec):
        """Prepare tracked persons for publishing (convert back for costmap publisher)."""
        
        # Convert to dictionary format expected by SocialCostmapPublisher
        published_persons = []
        for t in tracks:
            p_dict = {
                'id': t.id,
                'position_3d_map': t.position,
                'confidence': t.confidence,
                'source': t.source
            }
            published_persons.append(p_dict)
            
        # Create header stamp
        ts_ros = rclpy.time.Time(seconds=timestamp_sec).to_msg()
        
        # Publish PointCloud
        point_cloud = self.costmap_publisher_module.create_social_pointcloud(
            persons=published_persons,
            map_frame='map',
            timestamp=ts_ros
        )
        self.social_pub.publish(point_cloud)
        
        # Publish Markers
        markers = self.costmap_publisher_module.create_person_markers(
            persons=published_persons,
            map_frame='map',
            timestamp=ts_ros
        )
        self.markers_pub.publish(markers)

    
    def _create_registered_depth(self, cv_depth: np.ndarray, cv_rgb: np.ndarray, 
                                rgb_frame_id: str) -> Optional[np.ndarray]:
        """
        Create depth image registered (aligned) to RGB frame.
        
        Args:
            cv_depth: Raw depth image from depth camera
            cv_rgb: RGB image
            rgb_frame_id: RGB camera frame ID
        
        Returns:
            Registered depth image (same size as RGB) or None if failed
        """
        try:
            # Get transform from Depth Frame to RGB Frame
            transform = self.tf_buffer.lookup_transform(
                self.rgb_camera_model.header.frame_id,
                self.depth_camera_model.header.frame_id,
                rclpy.time.Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF Error (depth to rgb): {e}')
            return None

        # Camera Intrinsics
        depth_K = np.array(self.depth_camera_model.k).reshape(3, 3)
        rgb_K = np.array(self.rgb_camera_model.k).reshape(3, 3)
        
        # Rotation and Translation from Depth to RGB
        q = transform.transform.rotation
        t = transform.transform.translation
        
        # Quaternion to Rotation Matrix
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        
        R_depth_to_rgb = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
        ])
        
        T_depth_to_rgb = np.array([t.x, t.y, t.z])
        
        # Vectorized depth registration
        rgb_height, rgb_width = cv_rgb.shape[:2]
        depth_height, depth_width = cv_depth.shape
        
        u_grid, v_grid = np.meshgrid(np.arange(depth_width), np.arange(depth_height))
        
        depth_fx = depth_K[0, 0]
        depth_fy = depth_K[1, 1]
        depth_cx = depth_K[0, 2]
        depth_cy = depth_K[1, 2]
        
        Z_depth = cv_depth
        X_depth = (u_grid - depth_cx) * Z_depth / depth_fx
        Y_depth = (v_grid - depth_cy) * Z_depth / depth_fy
        
        # Flatten and filter valid depths
        valid_mask = (Z_depth > 0) & (~np.isnan(Z_depth))
        points_depth = np.vstack((X_depth[valid_mask], Y_depth[valid_mask], Z_depth[valid_mask])).T
        
        # Transform to RGB Frame
        points_rgb_3d = np.dot(points_depth, R_depth_to_rgb.T) + T_depth_to_rgb
        
        # Project to RGB Image
        rgb_fx = rgb_K[0, 0]
        rgb_fy = rgb_K[1, 1]
        rgb_cx = rgb_K[0, 2]
        rgb_cy = rgb_K[1, 2]
        
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
        
        return registered_depth
    
    def _save_debug_data(self, cv_rgb: np.ndarray, registered_depth: np.ndarray,
                        detections: List[Dict], localized_persons: List[Dict]):
        """Save debug images and data to disk."""
        try:
            os.makedirs(self.debug_dir, exist_ok=True)
            
            # Save annotated RGB with detections
            annotated_rgb = self.detector.visualize_detections(
                cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2RGB),
                detections,
                show_masks=True
            )
            cv2.imwrite(
                os.path.join(self.debug_dir, 'detected_persons.jpg'),
                cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            )
            
            # Save registered depth visualization
            depth_vis = cv2.normalize(registered_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.debug_dir, 'registered_depth.png'), depth_colormap)
            
            # Save person data to text file
            with open(os.path.join(self.debug_dir, 'persons.txt'), 'w') as f:
                f.write(f'Detected {len(localized_persons)} person(s)\n\n')
                for i, person in enumerate(localized_persons):
                    f.write(f'Person {i+1}:\n')
                    f.write(f'  Confidence: {person["confidence"]:.3f}\n')
                    if 'position_3d_camera' in person:
                        pos_cam = person['position_3d_camera']
                        f.write(f'  Camera Frame: ({pos_cam[0]:.3f}, {pos_cam[1]:.3f}, {pos_cam[2]:.3f})\n')
                    if 'position_3d_map' in person:
                        pos_map = person['position_3d_map']
                        f.write(f'  Map Frame: ({pos_map[0]:.3f}, {pos_map[1]:.3f}, {pos_map[2]:.3f})\n')
                    if 'depth_samples' in person:
                        f.write(f'  Depth samples: {person["depth_samples"]}\n')
                    f.write('\n')
            
            # self.get_logger().info(f'Debug data saved to {self.debug_dir}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to save debug data: {e}')




def main(args=None):
    rclpy.init(args=args)
    social_costmap_node = SocialCostmapNode()
    rclpy.spin(social_costmap_node)
    social_costmap_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
