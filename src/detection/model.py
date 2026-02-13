"""
Product Detection Model Wrapper
"""
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
import cv2

class ProductDetector:
    
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45, device: str = None):
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load model
        self.model = self._load_model()
        
    def _load_model(self) -> YOLO:
        """Load YOLO model"""
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def detect(self, image_path: str) -> List[Dict]:
        
        # Run inference
        results = self.model.predict(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        detections = []
        result = results[0]
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = box
                
                detection = {
                    'id': i,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': 'product'
                }
                detections.append(detection)
        
        return detections
    
    def detect_batch(self, image_paths: List[str]) -> List[List[Dict]]:
        """
        Detect products in multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of detection lists
        """
        results = self.model.predict(
            image_paths,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        all_detections = []
        for result in results:
            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = box
                    detection = {
                        'id': i,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'class': 'product'
                    }
                    detections.append(detection)
            
            all_detections.append(detections)
        
        return all_detections
    
    def extract_crops(self, image_path: str, detections: List[Dict]) -> List[np.ndarray]:
        """
        Extract cropped regions for each detection
        
        Args:
            image_path: Path to input image
            detections: List of detection dictionaries
            
        Returns:
            List of cropped image arrays
        """
        image = cv2.imread(str(image_path))
        crops = []
        
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            
            # Ensure coordinates are within bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            crop = image[y1:y2, x1:x2]
            crops.append(crop)
        
        return crops
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'model_type': 'YOLOv8m'
        }

