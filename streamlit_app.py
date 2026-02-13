"""
Retail Product Analyzer – Streamlit Frontend
"""

import streamlit as st
import requests
from PIL import Image
import io
import base64
import pandas as pd

# ---------------------------
# Configuration
# ---------------------------
API_URL = "https://kashishkhangarot290--retail-shelf-analysis-fastapi-app.modal.run"

st.set_page_config(
    page_title="Retail Product Analyzer",
    page_icon="🛒",
    layout="wide"
)

# ---------------------------
# Header
# ---------------------------
st.markdown("""
# 🛒 Retail Product Analyzer
Analyze shelf images using AI-powered detection & grouping
""")

st.markdown("---")

# ---------------------------
# Layout Blocks
# ---------------------------
input_col, output_col = st.columns(2)

# ---------------------------
# INPUT BLOCK
# ---------------------------
with input_col:

    st.markdown("## 📥 Input")

    uploaded_file = st.file_uploader(
        "Upload Retail Shelf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        st.image(image, caption="Loaded Image", use_column_width=True)

        analyze_btn = st.button(
            "🚀 Run Analysis",
            use_container_width=True
        )

# ---------------------------
# OUTPUT BLOCK
# ---------------------------
with output_col:

    st.markdown("## 📤 Output")

    if uploaded_file and analyze_btn:

        progress = st.progress(0, text="Starting analysis...")

        try:
            progress.progress(25, text="Uploading image...")

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            progress.progress(50, text="Running model inference on GPU...")

            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                timeout=120
            )

            if response.status_code != 200:
                st.error(f"API Error {response.status_code}")
                st.stop()

            result = response.json()

            progress.progress(100, text="Analysis complete ✅")

            # ---------------------------
            # Metrics
            # ---------------------------
            m1, m2 = st.columns(2)

            m1.metric("Total Products", result.get("total_products", 0))
            m2.metric("Brand Groups", result.get("total_groups", 0))

            # ---------------------------
            # Visualization
            # ---------------------------
            viz_b64 = result.get("visualization_b64")

            if viz_b64:
                viz_bytes = base64.b64decode(viz_b64)
                viz_img = Image.open(io.BytesIO(viz_bytes))

                st.image(
                    viz_img,
                    caption="Detection Visualization",
                    use_column_width=True
                )

            # ---------------------------
            # Table
            # ---------------------------
            objects = result.get("objects", [])

            if objects:
                rows = []

                for i, obj in enumerate(objects):
                    rows.append({
                        "ID": i + 1,
                        "Group": obj.get("group_id", "N/A"),
                        "Confidence": f"{obj.get('confidence', 0):.3f}",
                        "Bounding Box": tuple(int(v) for v in obj["bbox"])
                    })

                df = pd.DataFrame(rows)

                st.dataframe(df, use_container_width=True)

                st.download_button(
                    "⬇ Download CSV",
                    df.to_csv(index=False),
                    file_name="detections.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            else:
                st.info("No products detected.")

        except requests.exceptions.Timeout:
            st.error("Request timed out.")
        except requests.exceptions.ConnectionError:
            st.error("Backend connection failed.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")

st.caption("""
Retail Product Analyzer • YOLOv8m • CLIP • FastAPI • Modal GPU
""")
