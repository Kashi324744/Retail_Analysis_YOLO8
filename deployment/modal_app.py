"""
Modal deployment for Retail Shelf Analysis using YOLOv8m
"""

import modal
import base64
from pathlib import Path
import os

# ---------------------------
# Modal App
# ---------------------------
app = modal.App("retail-shelf-analysis")

# ---------------------------
# Get the src directory path
# ---------------------------
src_path = str(Path(__file__).parent.parent / "src")

# ---------------------------
# Image with dependencies and source code
# ---------------------------
image = (
    modal.Image.debian_slim()
    .apt_install(
        "libgl1",
        "libglib2.0-0"
    )
    .pip_install(
        "torch",
        "torchvision",
        "ultralytics",
        "opencv-python-headless",
        "pillow",
        "numpy",
        "fastapi",
        "python-multipart",
        "scikit-learn",
        "transformers==4.37.2",
    )
    .add_local_dir(src_path, remote_path="/root/src")
)

# ---------------------------
# Persistent Model Storage
# ---------------------------
model_volume = modal.Volume.from_name(
    "retail-models", create_if_missing=True
)

# ---------------------------
# GPU Inference Function
# ---------------------------
@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": model_volume},
    timeout=600,
)
def detect_products(image_b64: str):
    """
    Runs YOLOv8m inference on GPU using trained best.pt
    """
    import io
    import os
    import sys
    import cv2
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    
    # Add /root to sys.path so we can import src
    sys.path.insert(0, '/root')

    # Now import from src (which is at /root/src)
    from src.grouping.brand_grouper import BrandGrouper
    from src.visualization.visualizer import Visualizer

    # Decode input image
    image_bytes = base64.b64decode(image_b64)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(pil_image)

    # Load trained YOLOv8 model
    model_path = "/models/best.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        model = YOLO("yolov8m.pt")  # fallback

    model.to("cuda")

    # Run detection
    results = model.predict(
        image_np,
        conf=0.25,
        device="cuda",
        verbose=False,
    )

    detections = []
    result = results[0]

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        # Build detections
        for box, conf in zip(boxes, confs):
            detections.append({
                "bbox": [float(v) for v in box],
                "confidence": float(conf)
            })

    # ----------------------------
    # Real Brand Grouping
    # ----------------------------
    if detections:
        # Extract crops
        crops = []

        h, w = image_np.shape[:2]

        for det in detections:

            x1, y1, x2, y2 = map(int, det['bbox'])

            # ✅ Clamp bounding boxes (VERY IMPORTANT)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # ✅ Prevent empty / invalid crops
            if x2 > x1 and y2 > y1:

                crop = image_np[y1:y2, x1:x2]

                # ✅ RGB → BGR (YOUR MAIN FIX)
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

                crops.append(crop)


        # Use BrandGrouper
        grouper = BrandGrouper(similarity_threshold=0.85)
        group_ids = grouper.assign_groups(crops)
        from collections import Counter
        group_counts = Counter(group_ids)
        print(f"Group distribution: {dict(group_counts)}")
        print(f"Largest group: {max(group_counts.values())} items")
        print(f"Average group size: {len(group_ids) / len(set(group_ids)):.2f}")

        # Update detections
        for det, gid in zip(detections, group_ids):
            det['group_id'] = gid

        # ----------------------------
        # Visualization - FIXED
        # ----------------------------
        # Save the original image to a temporary file
        temp_input_path = "/tmp/input_image.jpg"
        # Use PIL to save (more reliable than cv2)
        pil_image.save(temp_input_path, "JPEG")
        
        # Verify the file was saved
        if not os.path.exists(temp_input_path):
            print(f"ERROR: Failed to save temporary image to {temp_input_path}")
            viz_b64 = None
        else:
            visualizer = Visualizer(line_width=2, font_size=0.5)
            output_path = "/tmp/output_image.jpg"
            
            try:
                # Pass the file path to visualizer
                visualizer.draw_detections(temp_input_path, detections, output_path)
                
                # Read the output and encode to base64
                with open(output_path, "rb") as f:
                    viz_b64 = base64.b64encode(f.read()).decode()
            except Exception as e:
                print(f"ERROR in visualization: {e}")
                viz_b64 = None
    else:
        viz_b64 = None

    # Count groups
    total_products = len(detections)
    unique_groups = len(set(det.get('group_id', 'unknown') for det in detections))

    return {
        "objects": detections,
        "visualization_b64": viz_b64,
        "total_products": total_products,
        "total_groups": unique_groups
    }


# ---------------------------
# FastAPI Service
# ---------------------------
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, '/root')
    
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    api = FastAPI(title="Retail Shelf Analysis API")
    
    # Add CORS
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class Detection(BaseModel):
        bbox: List[float]
        confidence: float
        group_id: Optional[str] = None

    class Response(BaseModel):
        objects: List[Detection]
        visualization_b64: Optional[str] = None
        total_products: int
        total_groups: int

    @api.get("/")
    async def root():
        """Root endpoint - health check"""
        return {
            "message": "Retail Shelf Analysis API is running on Modal!",
            "status": "healthy",
            "endpoints": {
                "predict": "/predict",
                "docs": "/docs"
            }
        }

    @api.post("/predict", response_model=Response)
    async def predict(file: UploadFile = File(...)):
        try:
            image_bytes = await file.read()
            image_b64 = base64.b64encode(image_bytes).decode()

            result = detect_products.remote(image_b64)

            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return api


# ---------------------------
# Local Test
# ---------------------------
@app.local_entrypoint()
def main(image_path: str):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    result = detect_products.remote(image_b64)

    print("Products:", result["total_products"])
    print("Groups:", result["total_groups"])
    print("Detections:", len(result["objects"]))
    
    if result["visualization_b64"]:
        # Save visualization locally
        viz_bytes = base64.b64decode(result["visualization_b64"])
        output_path = "local_output.jpg"
        with open(output_path, "wb") as f:
            f.write(viz_bytes)
        print(f"Visualization saved to {output_path}")
    else:
        print("No visualization was generated")