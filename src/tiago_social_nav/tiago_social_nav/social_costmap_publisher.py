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
    
    def create_social_pointcloud(self,
                                persons: List[Dict],
                                map_frame: str = 'map',
                                timestamp = None) -> PointCloud2:
        """
        Generate PointCloud2 with points representing social obstacles.
        
        Args:
            persons: List of localized persons with 'position_3d_map' field
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
            
            # 1. Physical body: fill with dense rings from center out to body radius
            # Create a solid obstacle for the person's body
            for radius in np.linspace(0.1, BODY_RADIUS, 3):
                num_points = 8
                for angle in np.linspace(0, 2*math.pi, num_points, endpoint=False):
                    x = px + radius * math.cos(angle)
                    y = py + radius * math.sin(angle)
                    z = 0.5  # fixed height
                    points.append((x, y, z))

            # 2. Intimate boundary ring at INTIMATE_EDGE
            # This marks the critical personal space boundary
            radius = INTIMATE_EDGE
            num_points = 24
            for angle in np.linspace(0, 2*math.pi, num_points, endpoint=False):
                x = px + radius * math.cos(angle)
                y = py + radius * math.sin(angle)
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

