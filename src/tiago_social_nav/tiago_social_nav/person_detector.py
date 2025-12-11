"""
YOLOv11n Segmentation-based Person Detector
Handles model loading, inference, and person detection filtering.
"""

import numpy as np
from typing import List, Dict
import cv2

from ultralytics import YOLO
from pathlib import Path
import shutil
import os

# Define cache directory relative to this file
# .../src/tiago_social_nav/tiago_social_nav/person_detector.py -> .../src/cache/ultralytics
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "ultralytics"



class PersonDetector:
    """Detect persons using YOLOv11n-seg model."""
    
    def __init__(self, 
                 model_name: str = 'yolo11n-seg.pt',
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = 'cuda'):
        """
        Initialize YOLO segmentation model.
        
        Args:
            model_name: YOLOv11 model variant (yolo11n-seg.pt for nano with segmentation)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: 'cuda' or 'cpu'
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load model with caching support
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_model_path = CACHE_DIR / model_name
        
        try:
            # Load from cache or download
            if cached_model_path.exists():
                self.model = YOLO(str(cached_model_path))
            else:
                # Download model, then move to cache if it landed in CWD
                self.model = YOLO(model_name)
                if Path(model_name).exists():
                    shutil.move(model_name, cached_model_path)
                    self.model = YOLO(str(cached_model_path))
            
            self.model.to(device)
        except Exception as e:
            # Fallback to CPU if GPU fails
            print(f"[PersonDetector] GPU load failed ({e}), falling back to CPU")
            self.device = 'cpu'
            if cached_model_path.exists():
                self.model = YOLO(str(cached_model_path))
            else:
                self.model = YOLO(model_name)
            self.model.to('cpu')
        
        # COCO class ID for 'person' is 0
        self.person_class_id = 0
    
    def detect(self, rgb_image: np.ndarray) -> List[Dict]:
        """
        Run YOLO inference on RGB image and return person detections.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
        
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - confidence: detection confidence score
                - mask: binary segmentation mask (H, W) or None if no mask
                - class_id: always 0 (person)
        """
        # Run inference
        results = self.model.predict(
            rgb_image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[self.person_class_id],  # Only detect persons
            verbose=False,
            device=self.device
        )
        
        detections = []
        
        if len(results) > 0:
            result = results[0]  # Single image inference
            
            # Check if we have detections
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confidences = result.boxes.conf.cpu().numpy()
                
                # Get masks if available
                masks = None
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()  # (N, H, W)
                
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes[i],
                        'confidence': float(confidences[i]),
                        'class_id': self.person_class_id,
                        'mask': masks[i] if masks is not None else None
                    }
                    detections.append(detection)
        
        return detections
