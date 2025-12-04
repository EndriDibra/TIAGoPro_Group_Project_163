"""
Social Costmap Publisher
Generates and publishes OccupancyGrid with social costs around detected persons.
"""

import numpy as np
from typing import List, Dict, Optional
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
import rclpy


class SocialCostmapPublisher:
    """Generate and publish social costmap as OccupancyGrid."""
    
    def __init__(self,
                 resolution: float = 0.05,
                 width: float = 10.0,
                 height: float = 10.0,
                 robot_radius: float = 0.3,
                 intimate_radius: float = 0.5,
                 personal_radius: float = 1.2,
                 social_radius: float = 3.6,
                 max_cost: int = 100):
        """
        Initialize social costmap publisher.
        
        Args:
            resolution: Grid resolution in meters per cell
            width: Costmap width in meters
            height: Costmap height in meters
            robot_radius: Robot radius for collision zone (meters)
            intimate_radius: Intimate zone radius - very high cost (meters)
            personal_radius: Personal zone radius - high cost (meters)
            social_radius: Social zone radius - moderate cost (meters)
            max_cost: Maximum cost value (0-100 for OccupancyGrid)
        """
        self.resolution = resolution
        self.width_meters = width
        self.height_meters = height
        self.width_cells = int(width / resolution)
        self.height_cells = int(height / resolution)
        
        # Social zone parameters
        self.robot_radius = robot_radius
        self.intimate_radius = intimate_radius
        self.personal_radius = personal_radius
        self.social_radius = social_radius
        self.max_cost = max_cost
        
        # Gaussian sigma for smooth cost falloff
        # Sigma chosen so that at zone boundary, cost is ~13% of max (2 sigma rule)
        self.intimate_sigma = intimate_radius / 2.0
        self.personal_sigma = personal_radius / 2.0
        self.social_sigma = social_radius / 2.0
    
    def create_costmap(self,
                      persons: List[Dict],
                      robot_position: Optional[np.ndarray] = None,
                      map_frame: str = 'map',
                      timestamp = None) -> OccupancyGrid:
        """
        Generate OccupancyGrid with social costs around persons.
        
        Args:
            persons: List of localized persons with 'position_3d_map' field
            robot_position: [x, y] robot position in map frame (for centering costmap)
            map_frame: Frame ID for the costmap
            timestamp: ROS timestamp for the message
        
        Returns:
            OccupancyGrid message
        """
        # Create empty grid
        grid = np.zeros((self.height_cells, self.width_cells), dtype=np.int8)
        
        # Determine costmap origin (center on robot if available, else at 0,0)
        if robot_position is not None:
            origin_x = robot_position[0] - self.width_meters / 2.0
            origin_y = robot_position[1] - self.height_meters / 2.0
        else:
            origin_x = -self.width_meters / 2.0
            origin_y = -self.height_meters / 2.0
        
        # Add costs for each person
        for person in persons:
            if 'position_3d_map' not in person:
                continue
            
            pos_map = person['position_3d_map']
            person_x, person_y = pos_map[0], pos_map[1]
            
            # Convert to grid coordinates
            grid_x = int((person_x - origin_x) / self.resolution)
            grid_y = int((person_y - origin_y) / self.resolution)
            
            # Skip if person is outside costmap bounds
            if not (0 <= grid_x < self.width_cells and 0 <= grid_y < self.height_cells):
                continue
            
            # Add social zones using Gaussian cost function
            self._add_social_zones(grid, grid_x, grid_y)
        
        # Create OccupancyGrid message
        occupancy_grid = OccupancyGrid()
        
        if timestamp is not None:
            occupancy_grid.header.stamp = timestamp
        occupancy_grid.header.frame_id = map_frame
        
        occupancy_grid.info.resolution = self.resolution
        occupancy_grid.info.width = self.width_cells
        occupancy_grid.info.height = self.height_cells
        
        # Set origin (lower-left corner of grid)
        occupancy_grid.info.origin.position.x = origin_x
        occupancy_grid.info.origin.position.y = origin_y
        occupancy_grid.info.origin.position.z = 0.0
        occupancy_grid.info.origin.orientation.w = 1.0
        
        # Flatten grid row-major (y-axis increases upward)
        occupancy_grid.data = grid.flatten().tolist()
        
        return occupancy_grid
    
    def _add_social_zones(self, grid: np.ndarray, center_x: int, center_y: int):
        """
        Add three-zone social cost model around a person.
        
        Uses concentric zones:
        1. Intimate zone (0-0.5m): max_cost (100)
        2. Personal zone (0.5-1.2m): high cost (70-100)
        3. Social zone (1.2-3.6m): moderate cost (30-70)
        
        Costs fall off with Gaussian function for smooth transitions.
        """
        height, width = grid.shape
        
        # Create meshgrid for distance calculation
        y_coords, x_coords = np.ogrid[0:height, 0:width]
        
        # Calculate distance from person center (in cells)
        dist_cells = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Convert to meters
        dist_meters = dist_cells * self.resolution
        
        # Three-zone cost model with Gaussian falloff
        cost = np.zeros_like(dist_meters)
        
        # Intimate zone: Highest cost (max within intimate_radius)
        intimate_mask = dist_meters <= self.intimate_radius
        cost[intimate_mask] = self.max_cost
        
        # Personal zone: High cost with Gaussian falloff
        personal_mask = (dist_meters > self.intimate_radius) & (dist_meters <= self.personal_radius)
        if np.any(personal_mask):
            # Gaussian centered at intimate boundary, sigma = personal_sigma
            dist_from_intimate = dist_meters[personal_mask] - self.intimate_radius
            gaussian_personal = np.exp(-(dist_from_intimate**2) / (2 * self.personal_sigma**2))
            cost[personal_mask] = self.max_cost * 0.7 + (self.max_cost * 0.3) * gaussian_personal
        
        # Social zone: Moderate cost with Gaussian falloff
        social_mask = (dist_meters > self.personal_radius) & (dist_meters <= self.social_radius)
        if np.any(social_mask):
            dist_from_personal = dist_meters[social_mask] - self.personal_radius
            gaussian_social = np.exp(-(dist_from_personal**2) / (2 * self.social_sigma**2))
            cost[social_mask] = self.max_cost * 0.3 + (self.max_cost * 0.4) * gaussian_social
        
        # Clip to valid range [0, 100] and convert to int8
        cost = np.clip(cost, 0, self.max_cost).astype(np.int8)
        
        # Merge with existing grid (take maximum cost)
        grid[:, :] = np.maximum(grid, cost)
    
    def create_person_markers(self,
                             persons: List[Dict],
                             map_frame: str = 'map',
                             timestamp = None,
                             namespace: str = 'detected_persons') -> MarkerArray:
        """
        Create visualization markers for detected persons.
        
        Args:
            persons: List of localized persons
            map_frame: Frame ID for markers
            timestamp: ROS timestamp
            namespace: Marker namespace
        
        Returns:
            MarkerArray with sphere markers at person positions
        """
        marker_array = MarkerArray()
        
        for i, person in enumerate(persons):
            if 'position_3d_map' not in person:
                continue
            
            pos_map = person['position_3d_map']
            
            # Create sphere marker
            marker = Marker()
            marker.header.frame_id = map_frame
            if timestamp is not None:
                marker.header.stamp = timestamp
            
            marker.ns = namespace
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = float(pos_map[0])
            marker.pose.position.y = float(pos_map[1])
            marker.pose.position.z = float(pos_map[2])
            marker.pose.orientation.w = 1.0
            
            # Scale (person size)
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 1.7  # Average human height
            
            # Color (green, semi-transparent)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.7
            
            # Lifetime
            marker.lifetime.sec = 1  # 1 second (will update at 5Hz)
            
            marker_array.markers.append(marker)
        
        # Add delete marker for any previously published markers beyond current count
        # This cleans up markers when persons leave the scene
        if len(persons) < 10:  # Assume max 10 tracked persons
            for i in range(len(persons), 10):
                delete_marker = Marker()
                delete_marker.header.frame_id = map_frame
                if timestamp is not None:
                    delete_marker.header.stamp = timestamp
                delete_marker.ns = namespace
                delete_marker.id = i
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)
        
        return marker_array
