"""
Person 3D Localization using Segmentation Masks and Depth Data
Converts detected person masks to 3D positions in map frame.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2


class PersonLocalizer:
    """Convert person detections to 3D positions using depth and segmentation."""
    
    def __init__(self, method: str = 'centroid'):
        """
        Initialize localizer.
        
        Args:
            method: '3d_centroid' or 'min_z_column' for depth sampling strategy
                - 3d_centroid: Compute weighted centroid from valid depth samples in mask
                - min_z_column: Find minimum depth per column, mirror on x-axis (more precise edges)
        """
        self.method = method
        if method not in ['3d_centroid', 'min_z_column']:
            raise ValueError(f"Unknown method: {method}. Use '3d_centroid' or 'min_z_column'")
    
    def localize_persons(self,
                        detections: List[Dict],
                        registered_depth: np.ndarray,
                        depth_K: np.ndarray,
                        depth_to_map_transform: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Convert person detections to 3D positions.
        
        Args:
            detections: List of detections from PersonDetector
            registered_depth: Registered depth image aligned to RGB (H, W) in meters
            depth_K: Camera intrinsic matrix (3, 3)
            depth_to_map_transform: 4x4 transform matrix from depth camera to map frame (optional)
        
        Returns:
            List of localized persons with added fields:
                - position_3d_camera: [x, y, z] in camera frame (meters)
                - position_3d_map: [x, y, z] in map frame (meters) if transform provided
                - depth_samples: number of valid depth samples used
        """
        localized_persons = []
        
        for det in detections:
            mask = det.get('mask')
            bbox = det['bbox']
            
            if mask is None:
                # Fallback to bounding box center if no mask
                position_camera = self._bbox_center_depth(bbox, registered_depth, depth_K)
            else:
                # Use mask-based method
                if self.method == '3d_centroid':
                    position_camera, num_samples = self._centroid_method(
                        mask, registered_depth, depth_K)
                else:  # min_z_column
                    position_camera, num_samples = self._min_z_column_method(
                        mask, registered_depth, depth_K)
                
                det['depth_samples'] = num_samples
            
            if position_camera is None:
                continue  # Skip if no valid depth
            
            det['position_3d_camera'] = position_camera
            
            # Transform to map frame if transform provided
            if depth_to_map_transform is not None:
                position_map = self._transform_to_map(position_camera, depth_to_map_transform)
                det['position_3d_map'] = position_map
            
            localized_persons.append(det)
        
        return localized_persons
    
    def _centroid_method(self,
                        mask: np.ndarray,
                        registered_depth: np.ndarray,
                        depth_K: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """
        Compute 3D centroid of person using mask and depth.
        
        Returns:
            (position_3d, num_samples) or (None, 0) if no valid depth
        """
        # Resize mask to match depth image if needed
        if mask.shape != registered_depth.shape:
            mask = cv2.resize(mask, (registered_depth.shape[1], registered_depth.shape[0]))
        
        # Get mask pixels
        mask_bool = mask > 0.5
        
        # Get valid depth values within mask
        valid_depth_mask = mask_bool & (registered_depth > 0) & (~np.isnan(registered_depth))
        
        if not np.any(valid_depth_mask):
            return None, 0
        
        # Get pixel coordinates and depths
        v_coords, u_coords = np.where(valid_depth_mask)
        depths = registered_depth[valid_depth_mask]
        
        # Deproject to 3D camera frame
        fx, fy = depth_K[0, 0], depth_K[1, 1]
        cx, cy = depth_K[0, 2], depth_K[1, 2]
        
        X = (u_coords - cx) * depths / fx
        Y = (v_coords - cy) * depths / fy
        Z = depths
        
        # Compute centroid (median is more robust to outliers than mean)
        position_3d = np.array([
            np.median(X),
            np.median(Y),
            np.median(Z)
        ])
        
        return position_3d, len(depths)
    
    def _min_z_column_method(self,
                            mask: np.ndarray,
                            registered_depth: np.ndarray,
                            depth_K: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """
        Find minimum depth per column in mask, mirror on x-axis for precise person boundary.
        This gives better edge detection for tight spaces.
        
        Returns:
            (position_3d, num_samples) or (None, 0) if no valid depth
        """
        # Resize mask to match depth image if needed
        if mask.shape != registered_depth.shape:
            mask = cv2.resize(mask, (registered_depth.shape[1], registered_depth.shape[0]))
        
        mask_bool = mask > 0.5
        
        # Get valid depth mask
        valid_depth_mask = mask_bool & (registered_depth > 0) & (~np.isnan(registered_depth))
        
        if not np.any(valid_depth_mask):
            return None, 0
        
        # For each column (u coordinate), find minimum depth
        height, width = registered_depth.shape
        fx, fy = depth_K[0, 0], depth_K[1, 1]
        cx, cy = depth_K[0, 2], depth_K[1, 2]
        
        min_depth_points = []
        
        for u in range(width):
            column_mask = valid_depth_mask[:, u]
            if np.any(column_mask):
                v_coords = np.where(column_mask)[0]
                depths = registered_depth[column_mask, u]
                
                # Find minimum depth in this column
                min_idx = np.argmin(depths)
                v = v_coords[min_idx]
                z = depths[min_idx]
                
                # Deproject to 3D
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                
                min_depth_points.append([x, y, z])
        
        if len(min_depth_points) == 0:
            return None, 0
        
        min_depth_points = np.array(min_depth_points)
        
        # Mirror on x-axis: add mirrored points to get symmetric boundary
        mirrored_points = min_depth_points.copy()
        mirrored_points[:, 0] *= -1  # Mirror x coordinates
        
        all_points = np.vstack([min_depth_points, mirrored_points])
        
        # Compute centroid of all boundary points
        position_3d = np.median(all_points, axis=0)
        
        return position_3d, len(min_depth_points)
    
    def _bbox_center_depth(self,
                          bbox: np.ndarray,
                          registered_depth: np.ndarray,
                          depth_K: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback: Get depth at bounding box center.
        
        Returns:
            position_3d or None if no valid depth
        """
        x1, y1, x2, y2 = bbox.astype(int)
        center_u = (x1 + x2) // 2
        center_v = (y1 + y2) // 2
        
        # Clamp to image bounds
        center_u = np.clip(center_u, 0, registered_depth.shape[1] - 1)
        center_v = np.clip(center_v, 0, registered_depth.shape[0] - 1)
        
        # Get depth at center (with small neighborhood for robustness)
        neighborhood_size = 5
        v1 = max(0, center_v - neighborhood_size)
        v2 = min(registered_depth.shape[0], center_v + neighborhood_size + 1)
        u1 = max(0, center_u - neighborhood_size)
        u2 = min(registered_depth.shape[1], center_u + neighborhood_size + 1)
        
        depth_patch = registered_depth[v1:v2, u1:u2]
        valid_depths = depth_patch[(depth_patch > 0) & (~np.isnan(depth_patch))]
        
        if len(valid_depths) == 0:
            return None
        
        z = np.median(valid_depths)
        
        # Deproject
        fx, fy = depth_K[0, 0], depth_K[1, 1]
        cx, cy = depth_K[0, 2], depth_K[1, 2]
        
        x = (center_u - cx) * z / fx
        y = (center_v - cy) * z / fy
        
        return np.array([x, y, z])
    
    def _transform_to_map(self,
                         position_camera: np.ndarray,
                         transform_matrix: np.ndarray) -> np.ndarray:
        """
        Transform 3D position from camera frame to map frame.
        
        Args:
            position_camera: [x, y, z] in camera frame
            transform_matrix: 4x4 homogeneous transformation matrix
        
        Returns:
            position_map: [x, y, z] in map frame
        """
        # Convert to homogeneous coordinates
        pos_homo = np.append(position_camera, 1.0)
        
        # Apply transform
        pos_map_homo = transform_matrix @ pos_homo
        
        # Convert back to 3D
        position_map = pos_map_homo[:3]
        
        return position_map
    
    @staticmethod
    def create_transform_matrix(translation: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Create 4x4 homogeneous transformation matrix.
        
        Args:
            translation: [x, y, z] translation vector
            rotation_matrix: 3x3 rotation matrix
        
        Returns:
            4x4 transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation
        return T
