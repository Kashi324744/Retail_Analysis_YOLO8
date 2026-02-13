"""
Visualization utilities for detected products and groups
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import random


class Visualizer:
    
    
    def __init__(self, line_width: int = 2, font_size: float = 0.5):
        
        self.line_width = line_width
        self.font_size = font_size
        self.color_map = {}
    
    def _get_color_for_group(self, group_id: str) -> Tuple[int, int, int]:
        
        if group_id not in self.color_map:
            # Generate random but vibrant color
            random.seed(hash(group_id))
            self.color_map[group_id] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
        
        return self.color_map[group_id]
    
    def draw_detections(self, image_path: str, detections: List[Dict], 
                       output_path: str) -> str:
       
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Draw each detection
        for det in detections:
            bbox = det['bbox']
            group_id = det.get('group_id', 'unknown')
            confidence = det.get('confidence', 0.0)
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Get color for this group
            color = self._get_color_for_group(group_id)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_width)
            
            # Prepare label
            label = f"{group_id} ({confidence:.2f})"
            
            # Calculate label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, 1
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_size,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA
            )
        
        # Add summary text
        summary = f"Total Products: {len(detections)} | Groups: {len(set(d.get('group_id', 'unknown') for d in detections))}"
        cv2.putText(
            image,
            summary,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        # Save output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        
        return str(output_path)
    
    def create_group_visualization(self, image_path: str, detections: List[Dict],
                                   output_path: str) -> str:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Group detections by group_id
        groups = {}
        for det in detections:
            group_id = det.get('group_id', 'unknown')
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(det)
        
        # Draw each group with distinct color
        for group_id, group_dets in groups.items():
            color = self._get_color_for_group(group_id)
            
            for det in group_dets:
                bbox = det['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Draw thicker box for grouped items
                cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_width + 1)
        
        # Add legend
        y_offset = 50
        for i, (group_id, group_dets) in enumerate(groups.items()):
            color = self._get_color_for_group(group_id)
            text = f"{group_id}: {len(group_dets)} items"
            
            # Draw legend box
            cv2.rectangle(image, (10, y_offset), (30, y_offset + 20), color, -1)
            cv2.putText(
                image, text, (40, y_offset + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
            y_offset += 30
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        
        return str(output_path)
    
    def reset_colors(self):
        """Reset color map"""
        self.color_map = {}
