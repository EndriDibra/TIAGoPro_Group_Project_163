
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.optimize import linear_sum_assignment

@dataclass
class TrackedPerson:
    id: int
    position: np.ndarray  # [x, y, z] in map frame
    velocity: np.ndarray  # [vx, vy, vz] in map frame
    last_update_time: float
    last_yolo_time: float  # Last time YOLO confirmed this person
    confidence: float
    source: str  # 'yolo', 'laser', 'fused'
    
    # Velocity smoothing factor
    VELOCITY_ALPHA: float = 0.3
    
    def predict(self, current_time: float) -> np.ndarray:
        """Predict position at current_time based on velocity (constant velocity model)."""
        dt = current_time - self.last_update_time
        return self.position + self.velocity * dt

    def update(self, measurement: np.ndarray, source: str, current_time: float):
        """Update state with new measurement."""
        # Update velocity with exponential smoothing
        dt = current_time - self.last_update_time
        if dt > 0.01:  # Avoid division by tiny dt
            new_velocity = (measurement - self.position) / dt
            self.velocity = self.VELOCITY_ALPHA * new_velocity + (1 - self.VELOCITY_ALPHA) * self.velocity
        
        self.position = measurement
        self.last_update_time = current_time
        self.source = source
        
        # YOLO updates reset the YOLO confirmation time
        if source == 'yolo':
            self.last_yolo_time = current_time

class PersonTracker:
    # Confidence decays from 1.0 to 0.0 over this many seconds without YOLO confirmation
    CONFIDENCE_DECAY_TIME: float = 20.0
    
    def __init__(self, 
                 decay_time: float = 20.0, 
                 distance_threshold: float = 0.5):
        self.tracks: List[TrackedPerson] = []
        self.next_id = 0
        self.decay_time = decay_time
        self.distance_threshold = distance_threshold
    
    def update_tracks(self, 
                     yolo_detections: List[np.ndarray], 
                     laser_clusters: List[np.ndarray], 
                     current_time: float):
        """
        Main update loop.
        1. Match YOLO detections to existing tracks using Hungarian algorithm.
        2. Create new tracks for unmatched YOLO detections.
        3. Match Laser clusters to existing tracks (that weren't updated by YOLO this frame).
        4. Apply time-based confidence decay and remove dead tracks.
        """
        
        matched_track_indices = set()
        unmatched_yolo = list(range(len(yolo_detections)))
        
        # 1. Hungarian Algorithm Matching for YOLO detections
        if len(yolo_detections) > 0 and len(self.tracks) > 0:
            # Build cost matrix (distance between each detection and track)
            cost_matrix = np.zeros((len(yolo_detections), len(self.tracks)))
            for i, yolo_pos in enumerate(yolo_detections):
                for j, track in enumerate(self.tracks):
                    cost_matrix[i, j] = np.linalg.norm(track.position - yolo_pos)
            
            # Run Hungarian algorithm for optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            unmatched_yolo = []
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < self.distance_threshold:
                    # Valid match - update track
                    self.tracks[j].update(
                        measurement=yolo_detections[i],
                        source='yolo',
                        current_time=current_time
                    )
                    matched_track_indices.add(j)
                else:
                    # Distance too large - treat as unmatched
                    unmatched_yolo.append(i)
            
            # Add detections that weren't assigned at all
            assigned_detections = set(row_ind)
            for i in range(len(yolo_detections)):
                if i not in assigned_detections:
                    unmatched_yolo.append(i)
        
        # 2. Create new tracks for unmatched YOLO detections
        for idx in unmatched_yolo:
            yolo_pos = yolo_detections[idx]
            new_track = TrackedPerson(
                id=self.next_id,
                position=yolo_pos,
                velocity=np.zeros(3),
                last_update_time=current_time,
                last_yolo_time=current_time,
                confidence=1.0,
                source='yolo'
            )
            self.tracks.append(new_track)
            self.next_id += 1
            
        # 3. Update remaining tracks with Laser
        # Only consider tracks that were NOT updated by YOLO this frame
        if len(laser_clusters) > 0:
            for j, track in enumerate(self.tracks):
                if j in matched_track_indices:
                    continue  # Already updated by YOLO
                
                # Find closest laser cluster using predicted position
                predicted_pos = track.predict(current_time)
                
                best_cluster_idx = -1
                min_dist = self.distance_threshold
                
                for i, cluster_pos in enumerate(laser_clusters):
                    # Compare 2D distance (laser is on a horizontal plane)
                    dist = np.linalg.norm(predicted_pos[:2] - cluster_pos[:2])
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster_idx = i
                
                if best_cluster_idx != -1:
                    cluster_pos = laser_clusters[best_cluster_idx]
                    # Keep Z from previous estimate since laser is flat
                    updated_pos = np.array([cluster_pos[0], cluster_pos[1], track.position[2]])
                    track.update(
                        measurement=updated_pos,
                        source='laser',
                        current_time=current_time
                    )

        # 4. Apply unified time-based confidence decay and prune dead tracks
        active_tracks = []
        for track in self.tracks:
            # Confidence decays based on time since last YOLO confirmation
            time_since_yolo = current_time - track.last_yolo_time
            track.confidence = max(0.0, 1.0 - time_since_yolo / self.CONFIDENCE_DECAY_TIME)
            
            # Keep track if confidence is above threshold
            if track.confidence > 0.05:
                active_tracks.append(track)
            
        self.tracks = active_tracks
        
        return self.tracks
