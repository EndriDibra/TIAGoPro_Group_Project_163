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
CACHE_DIR = Path(__file__).parents[2] / "cache" / "ultralytics"



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
        
        # Load model (will download on first run)
        # Load model using cache
        try:
            # Ensure cache dir exists
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            cached_model_path = CACHE_DIR / model_name
            
            if not cached_model_path.exists():
                print(f"[PersonDetector] Model not found in cache. Downloading {model_name}...")
                # Download to current directory first (YOLO default)
                model = YOLO(model_name) 
                
                # Move to cache if it was downloaded to CWD
                if Path(model_name).exists():
                    print(f"[PersonDetector] Moving {model_name} to {CACHE_DIR}...")
                    shutil.move(model_name, cached_model_path)
                    # Reload from new location
                    self.model = YOLO(str(cached_model_path))
                else:
                    # It might have been downloaded elsewhere or we can't find it. 
                    # Just use the model object we have, but warn.
                    print(f"[PersonDetector] formatted model path not found in CWD, using loaded model.")
                    self.model = model
            else:
                 print(f"[PersonDetector] Loading {model_name} from cache: {cached_model_path}")
                 self.model = YOLO(str(cached_model_path))

            self.model.to(device)
            print(f"[PersonDetector] Loaded {model_name} on {device}")
        except Exception as e:
            print(f"[PersonDetector] Failed to load on {device}, falling back to CPU: {e}")
            self.device = 'cpu'
            # Fallback logic for CPU
            try:
                cached_model_path = CACHE_DIR / model_name
                if cached_model_path.exists():
                     self.model = YOLO(str(cached_model_path))
                else:
                     self.model = YOLO(model_name)
                self.model.to('cpu')
            except Exception as e2:
                print(f"[PersonDetector] CRITICAL: Failed to load model on CPU: {e2}")
                raise e2
        
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
    
    def visualize_detections(self, 
                            rgb_image: np.ndarray, 
                            detections: List[Dict],
                            show_masks: bool = True) -> np.ndarray:
        """
        Draw detections on RGB image for visualization.
        
        Args:
            rgb_image: Original RGB image
            detections: List of detections from detect()
            show_masks: Whether to overlay segmentation masks
        
        Returns:
            Annotated image
        """
        vis_image = rgb_image.copy()
        
        for det in detections:
            # Draw bounding box
            x1, y1, x2, y2 = det['bbox'].astype(int)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"Person: {det['confidence']:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw mask overlay
            if show_masks and det['mask'] is not None:
                mask = det['mask']
                # Resize mask to image size if needed
                if mask.shape != rgb_image.shape[:2]:
                    mask = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]))
                
                # Create colored overlay
                mask_bool = mask > 0.5
                overlay = vis_image.copy()
                overlay[mask_bool] = overlay[mask_bool] * 0.6 + np.array([0, 255, 0]) * 0.4
                vis_image = overlay.astype(np.uint8)
        
        return vis_image
