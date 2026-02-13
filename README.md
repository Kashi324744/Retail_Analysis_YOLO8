Retail Shelf Analysis System
AI-powered product detection and brand grouping for retail shelf images using YOLOv8 and CLIP.

Overview:
This system analyzes retail shelf images to:

Detect individual products using YOLOv8m
Group similar products by brand using CLIP embeddings
Visualize results with color-coded bounding boxes
Serve predictions via FastAPI deployed on Modal GPU

Dataset: SKU-110K (11,762 retail shelf images)
Model: YOLOv8m fine-tuned on SKU-110K
Deployment: Modal.com with T4 GPU
Key Results:

Detection mAP@0.5: 0.612
Inference time: ~190ms (GPU)
Handles 100+ products per image


Tech Stack:
Core Libraries
ComponentLibraryVersionPurposeDetectionultralytics8.0+YOLOv8 object detectionGroupingtransformers4.37.2CLIP visual embeddingsClusteringscikit-learn1.3+Agglomerative clusteringDeep Learningtorch2.0+PyTorch frameworkComputer Visionopencv-python4.8+Image processingAPIfastapi0.104+REST API frameworkFrontendstreamlit1.28+Web interfaceDeploymentmodal0.57+Serverless GPU platform
Full Dependencies
txttorch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
transformers==4.37.2
opencv-python-headless>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
fastapi>=0.104.0
python-multipart>=0.0.6
streamlit>=1.28.0
pandas>=2.0.0
modal>=0.57.0
requests>=2.31.0

Architecture
System Flow
┌─────────────────────────────────────────────────────────────┐
│                    USER (Streamlit UI)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ HTTP POST /predict
                           │ (multipart/form-data)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              MODAL CLOUD (T4 GPU)                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                FastAPI Endpoint                       │  │
│  │  • Receives image upload                              │  │
│  │  • Validates & decodes                                │  │
│  └─────────────────────┬─────────────────────────────────┘  │
│                        │                                     │
│                        ▼                                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         STEP 1: Product Detection                     │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  YOLOv8m Model                                  │  │  │
│  │  │  • Input: 640×640 image                         │  │  │
│  │  │  • Output: Bounding boxes + confidence          │  │  │
│  │  │  • Time: ~45ms                                  │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                        │                                     │
│                        ▼                                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         STEP 2: Crop Extraction                       │  │
│  │  • Extract each product region                        │  │
│  │  • Clamp to image boundaries                          │  │
│  │  • Convert RGB→BGR                                    │  │
│  │  • Time: ~12ms                                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                        │                                     │
│                        ▼                                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         STEP 3: Brand Grouping                        │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  CLIP ViT-B/32                                  │  │  │
│  │  │  • Extract 512-dim embeddings                   │  │  │
│  │  │  • Compute cosine similarity                    │  │  │
│  │  │  • Time: ~98ms                                  │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  Agglomerative Clustering                       │  │  │
│  │  │  • Distance threshold: 0.15                     │  │  │
│  │  │  • Linkage: average                             │  │  │
│  │  │  • Time: ~8ms                                   │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                        │                                     │
│                        ▼                                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         STEP 4: Visualization                         │  │
│  │  • Assign colors to groups                            │  │
│  │  • Draw bounding boxes                                │  │
│  │  • Add labels (group_id + confidence)                 │  │
│  │  • Encode as Base64                                   │  │
│  │  • Time: ~25ms                                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                        │                                     │
│                        ▼                                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         JSON Response                                 │  │
│  │  {                                                     │  │
│  │    "objects": [...],                                  │  │
│  │    "visualization_b64": "...",                        │  │
│  │    "total_products": N,                               │  │
│  │    "total_groups": M                                  │  │
│  │  }                                                     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    USER (Streamlit UI)                       │
│  • Display annotated image                                  │
│  • Show metrics (products, groups)                          │
│  • Export CSV                                               │
└─────────────────────────────────────────────────────────────┘

Code Structure:
retail-shelf-analysis/
├── src/
│   ├── detection/
│   │   ├── __init__.py
│   │   └── model.py              # ProductDetector (YOLOv8)
│   ├── grouping/
│   │   ├── __init__.py
│   │   └── brand_grouper.py      # BrandGrouper (CLIP + clustering)
│   └── visualization/
│       ├── __init__.py
│       └── visualizer.py         # Visualizer (draw boxes)
├── deployment/
│   └── modal_app.py              # Modal deployment config
├── models/
│   └── best.pt                   # Trained YOLOv8m weights (49.6MB)
├── streamlit_app.py              # Web interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file

Deployment
Prerequisites

Python 3.10+
Modal account (sign up here)
Git

Step 1: Clone Repository
bashgit clone https://github.com/yourusername/retail-shelf-analysis.git
cd retail-shelf-analysis
Step 2: Install Dependencies
bash# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
Step 3: Setup Modal
bash# Install Modal CLI
pip install modal

# Authenticate (opens browser)
modal setup
Step 4: Upload Model to Modal
bash# Create volume for model storage
modal volume create retail-models

# Upload trained model
modal volume put retail-models models/best.pt /best.pt

# Verify upload
modal volume ls retail-models
# Expected output: best.pt (49.6 MiB)
Step 5: Deploy to Modal
bashcd deployment
modal deploy modal_app.py
Output:
✓ Created objects.
✓ Created mount /root/src
✓ Created volume mount /models  
✓ Created web function fastapi_app
  => https://yourusername--retail-shelf-analysis-fastapi-app.modal.run
Copy the deployment URL!
Step 6: Configure Streamlit
Edit streamlit_app.py (line 14):
pythonAPI_URL = "https://yourusername--retail-shelf-analysis-fastapi-app.modal.run"
Replace with your actual Modal deployment URL.
Step 7: Run Streamlit
bash# Go back to project root
cd ..

# Run Streamlit
streamlit run streamlit_app.py
Opens browser at: http://localhost:8501
Step 8: Test

Upload a retail shelf image
Click "Run Analysis"
View results:

Annotated image with colored boxes
Detection count & group count
Detailed table
Download CSV


Configuration:
Detection Parameters
Edit deployment/modal_app.py (line 92):
pythonresults = model.predict(
    image_np,
    conf=0.25,      # Confidence threshold (0.1-0.9)
    device="cuda",
    verbose=False,
)
Grouping Parameters
Edit deployment/modal_app.py (line 148):
pythongrouper = BrandGrouper(similarity_threshold=0.85)
# Lower (0.7) = more groups (stricter)
# Higher (0.95) = fewer groups (more lenient)
GPU Selection
Edit deployment/modal_app.py (line 52):
python@app.function(
    image=image,
    gpu="T4",       # Options: T4, A10G, A100
    volumes={"/models": model_volume},
    timeout=600,
)

Performance:
MetricValueDetection mAP@0.50.612Detection Precision0.784Detection Recall0.689YOLOv8 Inference45ms (GPU)CLIP Feature Extraction98ms (GPU)Total Pipeline~190ms (GPU)Cold Start~8-12 seconds (first request)Warm Inference~3-5 seconds
