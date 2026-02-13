"""
FastAPI backend for retail shelf analysis
Assignment-level implementation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from pathlib import Path
import shutil
import uuid

from src.detection.model import ProductDetector
from src.grouping.brand_grouper import BrandGrouper
from src.visualization.visualizer import Visualizer

# -------------------- App -------------------- #

app = FastAPI(title="Retail Shelf Analysis API")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
MODEL_PATH = "models/best.pt"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------- Load Models -------------------- #

detector = ProductDetector(
    model_path=MODEL_PATH,
    conf_threshold=0.25,
    iou_threshold=0.45
)

grouper = BrandGrouper(
    similarity_threshold=0.85
)

visualizer = Visualizer()

# -------------------- API -------------------- #

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image and returns detected products with brand grouping
    """

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    # Save uploaded image
    image_id = str(uuid.uuid4())
    image_path = UPLOAD_DIR / f"{image_id}.jpg"

    with image_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Step 1: Detect products
    detections = detector.detect(str(image_path))

    if len(detections) == 0:
        return {
            "objects": [],
            "visualization_path": ""
        }

    # Step 2: Crop detected products
    crops = detector.extract_crops(str(image_path), detections)

    # Step 3: Group by visual similarity
    group_ids = grouper.assign_groups(crops)

    for det, gid in zip(detections, group_ids):
        det["group_id"] = gid

    # Step 4: Visualization
    output_image = OUTPUT_DIR / f"{image_id}.jpg"
    visualizer.draw_detections(
        str(image_path),
        detections,
        str(output_image)
    )

    # Step 5: Prepare response
    response_objects = [
        {
            "bbox": det["bbox"],
            "confidence": det["confidence"],
            "group_id": det["group_id"]
        }
        for det in detections
    ]
    # Compute metrics
    total_products = len(detections)
    total_groups = len(set(group_ids)) if group_ids else 0

    return {
        "objects": response_objects,
        "visualization_path": str(output_image),
        "total_products": total_products,
        "total_groups": total_groups
    }