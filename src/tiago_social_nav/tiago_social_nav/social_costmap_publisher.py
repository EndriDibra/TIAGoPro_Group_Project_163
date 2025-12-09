"""
Social Costmap Publisher
Generates and publishes PointCloud2 with social obstacles around detected persons.
"""

import numpy as np
import math
from typing import List, Dict
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import rclpy

# Proxemics constants
BODY_RADIUS = 0.25       # m
INTIMATE_FROM_EDGE = 0.45  # m
PERSONAL_FROM_EDGE = 1.20  # m

INTIMATE_EDGE = BODY_RADIUS + INTIMATE_FROM_EDGE   # 0.70 m
PERSONAL_EDGE = BODY_RADIUS + PERSONAL_FROM_EDGE   # 1.45 m


class SocialCostmapPublisher:
    """Generate and publish social obstacles as PointCloud2."""
    
    def __init__(self):
        """Initialize social costmap publisher."""
        # No grid parameters needed anymore
        pass
    
    # Prediction parameters
    PREDICTION_TIME = 2.0  # seconds - how far ahead to predict
    MIN_SPEED_THRESHOLD = 0.1  # m/s - below this, use circular costmap
    
    def create_social_pointcloud(self,
                                persons: List[Dict],
                                map_frame: str = 'map',
                                timestamp = None) -> PointCloud2:
        """
        Generate PointCloud2 with points representing social obstacles.
        Includes forward prediction zone based on velocity.
        
        Args:
            persons: List of localized persons with 'position_3d_map' and optional 'velocity' fields
            map_frame: Frame ID for the point cloud
            timestamp: ROS timestamp for the message
        
        Returns:
            PointCloud2 message
        """
        points = []
        
        for p in persons:
            if 'position_3d_map' not in p:
                continue
                
            pos = p['position_3d_map']
            px, py = pos[0], pos[1]
            
            # Get velocity if available
            velocity = p.get('velocity', np.zeros(3))
            vx, vy = velocity[0], velocity[1]
            speed = math.sqrt(vx**2 + vy**2)
            
            # 1. Physical body: fill with dense rings from center out to body radius
            for radius in np.linspace(0.1, BODY_RADIUS, 3):
                num_points = 8
                for angle in np.linspace(0, 2*math.pi, num_points, endpoint=False):
                    x = px + radius * math.cos(angle)
                    y = py + radius * math.sin(angle)
                    z = 0.5
                    points.append((x, y, z))

            # 2. Intimate boundary ring at INTIMATE_EDGE
            radius = INTIMATE_EDGE
            num_points = 24
            for angle in np.linspace(0, 2*math.pi, num_points, endpoint=False):
                x = px + radius * math.cos(angle)
                y = py + radius * math.sin(angle)
                z = 0.5
                points.append((x, y, z))
            
            # 3. Forward prediction zone if moving
            if speed > self.MIN_SPEED_THRESHOLD:
                heading = math.atan2(vy, vx)
                prediction_dist = speed * self.PREDICTION_TIME
                
                # Generate points along forward arc (only ahead, not behind)
                # Create a teardrop shape extending forward
                num_distance_steps = max(3, int(prediction_dist / 0.3))  # Point every ~0.3m
                
                for dist_step in range(1, num_distance_steps + 1):
                    # Distance from current position (forward only)
                    dist = (dist_step / num_distance_steps) * prediction_dist
                    
                    # Lateral width tapers as we go further ahead
                    # At person: width = INTIMATE_EDGE
                    # At max prediction: width = BODY_RADIUS
                    taper_factor = 1.0 - (dist / prediction_dist) * 0.6
                    lateral_width = INTIMATE_EDGE * taper_factor
                    
                    # Center point of this slice
                    center_x = px + dist * math.cos(heading)
                    center_y = py + dist * math.sin(heading)
                    
                    # Generate arc of points at this distance
                    # Perpendicular direction
                    perp_heading = heading + math.pi / 2
                    
                    num_lateral_points = max(3, int(lateral_width / 0.15))
                    for lat_step in range(-num_lateral_points, num_lateral_points + 1):
                        lat_offset = (lat_step / num_lateral_points) * lateral_width
                        x = center_x + lat_offset * math.cos(perp_heading)
                        y = center_y + lat_offset * math.sin(perp_heading)
                        z = 0.5
                        points.append((x, y, z))

        # Create header
        header = Header()
        header.frame_id = map_frame
        if timestamp is not None:
            header.stamp = timestamp
            
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        pc2_msg = pc2.create_cloud(header, fields, points)
        return pc2_msg
    
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
            
            # Create sphere marker for body
            marker = Marker()
            marker.header.frame_id = map_frame
            if timestamp is not None:
                marker.header.stamp = timestamp
            
            marker.ns = namespace
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = float(pos_map[0])
            marker.pose.position.y = float(pos_map[1])
            marker.pose.position.z = 0.9  # Center of 1.8m tall cylinder
            marker.pose.orientation.w = 1.0
            
            # Scale
            marker.scale.x = BODY_RADIUS * 2
            marker.scale.y = BODY_RADIUS * 2
            marker.scale.z = 1.8 
            
            # Color (Red for body)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker.lifetime.sec = 1
            marker_array.markers.append(marker)
            
            # Create Intimate Zone Ring
            ring = Marker()
            ring.header.frame_id = map_frame
            if timestamp is not None:
                ring.header.stamp = timestamp
            
            ring.ns = namespace + "_intimate"
            ring.id = i
            ring.type = Marker.CYLINDER
            ring.action = Marker.ADD
            
            ring.pose.position.x = float(pos_map[0])
            ring.pose.position.y = float(pos_map[1])
            ring.pose.position.z = 0.05
            ring.pose.orientation.w = 1.0
            
            # Outer diameter = 2 * INTIMATE_EDGE
            ring.scale.x = INTIMATE_EDGE * 2
            ring.scale.y = INTIMATE_EDGE * 2
            ring.scale.z = 0.05
            
            # Color (Orange/Yellow for intimate zone)
            ring.color.r = 1.0
            ring.color.g = 0.6
            ring.color.b = 0.0
            ring.color.a = 0.4
            
            ring.lifetime.sec = 1
            marker_array.markers.append(ring)
            
            # Create Personal Zone Ring
            p_ring = Marker()
            p_ring.header.frame_id = map_frame
            if timestamp is not None:
                p_ring.header.stamp = timestamp
            
            p_ring.ns = namespace + "_personal"
            p_ring.id = i
            p_ring.type = Marker.CYLINDER
            p_ring.action = Marker.ADD
            
            p_ring.pose.position.x = float(pos_map[0])
            p_ring.pose.position.y = float(pos_map[1])
            p_ring.pose.position.z = 0.01
            p_ring.pose.orientation.w = 1.0
            
            p_ring.scale.x = PERSONAL_EDGE * 2.0
            p_ring.scale.y = PERSONAL_EDGE * 2.0
            p_ring.scale.z = 0.01
            
            # Color (Green/Blue for personal zone)
            p_ring.color.r = 0.0
            p_ring.color.g = 0.5
            p_ring.color.b = 1.0
            p_ring.color.a = 0.2
            
            p_ring.lifetime.sec = 1
            marker_array.markers.append(p_ring)
            
            # Create Velocity Arrow marker (if moving)
            velocity = person.get('velocity', np.zeros(3))
            vx, vy = velocity[0], velocity[1]
            speed = math.sqrt(vx**2 + vy**2)
            
            if speed > self.MIN_SPEED_THRESHOLD:
                arrow = Marker()
                arrow.header.frame_id = map_frame
                if timestamp is not None:
                    arrow.header.stamp = timestamp
                
                arrow.ns = namespace + "_velocity"
                arrow.id = i
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                
                # Arrow starts at person position
                arrow.pose.position.x = float(pos_map[0])
                arrow.pose.position.y = float(pos_map[1])
                arrow.pose.position.z = 1.0  # Above the person
                
                # Orientation from velocity direction
                heading = math.atan2(vy, vx)
                arrow.pose.orientation.z = math.sin(heading / 2)
                arrow.pose.orientation.w = math.cos(heading / 2)
                
                # Arrow length proportional to prediction distance
                prediction_dist = speed * self.PREDICTION_TIME
                arrow.scale.x = prediction_dist  # Arrow length
                arrow.scale.y = 0.1  # Arrow width
                arrow.scale.z = 0.1  # Arrow height
                
                # Color (Cyan for velocity)
                arrow.color.r = 0.0
                arrow.color.g = 1.0
                arrow.color.b = 1.0
                arrow.color.a = 0.8
                
                arrow.lifetime.sec = 1
                marker_array.markers.append(arrow)
        
        # Cleanup old markers
        if len(persons) < 10:
            for i in range(len(persons), 10):
                # Body
                delete_marker = Marker()
                delete_marker.header.frame_id = map_frame
                delete_marker.ns = namespace
                delete_marker.id = i
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)
                
                # Intimate
                delete_intimate = Marker()
                delete_intimate.header.frame_id = map_frame
                delete_intimate.ns = namespace + "_intimate"
                delete_intimate.id = i
                delete_intimate.action = Marker.DELETE
                marker_array.markers.append(delete_intimate)

                # Personal
                delete_personal = Marker()
                delete_personal.header.frame_id = map_frame
                delete_personal.ns = namespace + "_personal"
                delete_personal.id = i
                delete_personal.action = Marker.DELETE
                marker_array.markers.append(delete_personal)
        
        return marker_array

