"""
Brand Grouping using Visual Similarity
"""
import numpy as np
import torch
from PIL import Image
from typing import List, Dict
import cv2
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

class BrandGrouper:
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self._load_clip()
    
    def _load_clip(self):
        """Load CLIP model for visual embeddings"""
        from transformers import CLIPProcessor, CLIPModel
        model_name = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        print(f"CLIP model loaded on {self.device}")
    
    
    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        features = []
        
        with torch.no_grad():
            for crop in crops:
                # Convert BGR to RGB
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                
                # Process image
                inputs = self.processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get image features
                img_features = self.model.get_image_features(**inputs)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                
                features.append(img_features.cpu().numpy().flatten())
        
        return np.array(features)
    
    
    def group_by_similarity(self, features: np.ndarray) -> List[int]:
        if len(features) == 0:
            return []
        
        if len(features) == 1:
            return [0]
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(features)
        
        # Clip to [0, 1] range to avoid numerical errors
        similarity_matrix = np.clip(similarity_matrix, 0, 1)
        
        # Convert similarity to distance for clustering
        distance_matrix = 1 - similarity_matrix
        
        # Ensure non-negative distances
        distance_matrix = np.maximum(distance_matrix, 0)
        
        # Make diagonal exactly 0 (distance to self)
        np.fill_diagonal(distance_matrix, 0)
        
        # Use Agglomerative Clustering instead of DBSCAN
        # This works better when we want to control the distance threshold
        from sklearn.cluster import AgglomerativeClustering
    
        distance_threshold = 1 - self.similarity_threshold
        
        print(f"\n[DEBUG] Clustering Info:")
        print(f"  Similarity threshold: {self.similarity_threshold}")
        print(f"  Distance threshold: {distance_threshold:.4f}")
        print(f"  Mean distance: {distance_matrix[~np.eye(distance_matrix.shape[0], dtype=bool)].mean():.4f}")
        print(f"  Min distance: {distance_matrix[~np.eye(distance_matrix.shape[0], dtype=bool)].min():.4f}")
        print(f"  Max distance: {distance_matrix[~np.eye(distance_matrix.shape[0], dtype=bool)].max():.4f}")
        
        # Use AgglomerativeClustering with distance threshold
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        
        group_ids = clusterer.fit_predict(distance_matrix)
        
        print(f"  Number of clusters found: {len(np.unique(group_ids))}")
        
        return group_ids.tolist()
    
    def assign_groups(self, crops: List[np.ndarray]) -> List[str]:
        """
        Assign group IDs to crops
        
        Args:
            crops: List of cropped images
            
        Returns:
            List of group IDs as strings (e.g., 'brand_0', 'brand_1')
        """
        if len(crops) == 0:
            return []
        
        # Extract features
        features = self.extract_features(crops)
        
        # Group by similarity
        group_ids = self.group_by_similarity(features)
        
        # Format as brand IDs
        brand_ids = [f"brand_{gid}" for gid in group_ids]
        
        return brand_ids

