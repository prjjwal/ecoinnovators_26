# AI-Powered Rooftop PV Detection Pipeline | EcoInnovators Ideathon 2026

**Team:** 52_Saints  
**Challenge:** Governance-Ready Remote Verification of Rooftop Solar  
**License:** MIT License (Open Source Initiative Approved)

##  Project Overview
This repository contains a production-grade AI pipeline designed to remotely verify rooftop solar installations for the *PM Surya Ghar: Muft Bijli Yojana* scheme. Our solution addresses the core challenge of auditability and scalability by using **YOLOv8-OBB (Oriented Bounding Box)** models and a **multi-stage verification logic** that mimics human inspection.

### The key limitations of the project are inferior image quality using free resources like ESRI database. The model is primarily trained on high-res images at zoom level 21 (drone imagery) along with some lower-res images at zoom 18 (satellite imagery). However using ESRI to fetch images at the same zoom level 21 returns inferior images, rendering the model useless. Alternatively, we've implemented a strategy to somewhat negate that issue using several image enhancement techniques on zoom level 18 images. This also leads to slightly reduced performance since the model is not trained on images at this zoom level. Thus, in order to truly utilise all the capabilities of our model, using paid third party services to fetch high-res satellitel images/drone imagery is recommended.

### Key Features
*   **Precision Area Estimation:** Uses Oriented Bounding Boxes (OBB) instead of standard boxes to capture precise panel area ($m^2$), handling tilted and irregular arrays on Indian roofs.
*   **Audit-Ready Artifacts:** Generates high-resolution evidence images with overlay masks for every single site.
*   **Advanced Image Fetching:** Implements "Jitter Sampling" (fetching 5 slightly offset images per site) and "Auto-Focus" quality control to reject blurry or cloudy imagery automatically.
*   **Two-Stage Verification:** Strictly follows the 1200 sq.ft vs. 2400 sq.ft buffer zone logic as per the problem statement.
*   **Dominant Panel Selection:** Complies with FAQ #4 by identifying the single dominant panel cluster overlapping the buffer zone.

> **ℹ Note on Buffer Logic & Zoom Strategy:**
> *   **Full Context Inference:** The model fetches imagery at **Zoom Level 18** to match the native resolution of the training dataset (~0.6m/px). It detects panels across the entire available tile context (768x768px).
> *   **Strict Filtering:** Per challenge rules, `has_solar` is strictly set to `True` **only if** a detected panel geometrically intersects the 1200/2400 sq.ft buffer zone. Detections outside this zone are ignored for the final JSON decision but can be visualized in the debug artifacts.
> *   **Verification:** To verify the model's performance beyond the buffer, check `output_results/debug_views/`.

***

##  System Requirements

*   **OS:** Linux (Ubuntu 20.04+) or Windows 10/11
*   **Python:** 3.8+
*   **Hardware:** 
    *   Inference: CPU (Supported) or NVIDIA GPU (Recommended, 4GB+ VRAM)
    *   Internet connection required for fetching live satellite imagery.

### Dependencies
Install all required libraries using:
```bash
pip install -r requirements.txt
```

*Contents of `requirements.txt`:*
```text
ultralytics==8.3.0
opencv-python-headless
numpy
pandas
requests
shapely
openpyxl
```

***

##  How to Run

### 1. Setup Input Data
Place your input Excel file in the `input_data/` folder. It must contain columns: `sample_id`, `latitude`, `longitude`.
*   Example: `input_data/input.xlsx`

### 2. Run the Pipeline
Execute the main script. You can specify custom paths if needed.
```bash
python pipeline.py --input_dir input_data --output_dir output_results --model solar_model.pt
```

### 3. View Results
A sample results folder is generated on a random lat-lon dataset containing 90 test cases. These co-ordinates are not part of the training dataset and represent a production use case of our system. These sample results can be viewed under 'test_results/'
The pipeline will generate three key outputs:
1.  **`output_results/predictions.json`**: The mandatory JSON submission file.
2.  **`output_results/artifacts/`**: Audit images for verified sites (Green = Selected Panel, Yellow = Buffer).
3.  **`output_results/debug_views/`**: **(For Judges/Debugging)** Raw view of what the model saw.
    *   **Cyan Boxes:** All raw detections found by the model.
    *   **Red Circle:** 1200 sq.ft search zone.
    *   **Blue Circle:** 2400 sq.ft search zone.
    *   *Use this folder to verify that the model is detecting panels even if they fall outside the strict buffer.*

***

##  Methodology & Logic

### 1. Image Acquisition (The "Fetcher")
We do not rely on a single static image. Our fetcher queries **Esri World Imagery (Wayback)** using a **Jitter Strategy**:
*   It fetches the target coordinate + 4 neighbors (North, South, East, West ~1.5m offset).
*   It calculates a **Sharpness Score** (Laplacian Variance) for each.
*   It selects the sharpest, clearest image for inference.
*   **QC Filter:** Images with low contrast or excessive blur are flagged as `NOT_VERIFIABLE` immediately.

### 2. Detection Model (YOLOv8l-OBB)
We trained a **Large (l)** YOLOv8-OBB model.
*   **Why OBB?** Solar panels are rigid rotated rectangles. Segmentation masks are often "blobby" and inaccurate for area calculation. OBB gives mathematically precise $Area = Width \times Height$.
*   **Training Data:** Merged and standardized datasets from **Alfred Weber**, **LSGI547**, and **Piscinas y Tenistable** (Roboflow Universe).

### 3. Quantification Logic (The "Calculator")
To strictly follow Challenge FAQ #4:
1.  The model detects all panels in the image (Full Context).
2.  We create a virtual **Buffer Zone** (1200 or 2400 sq.ft circle).
3.  We calculate the **Geometric Intersection** of every panel with this circle.
4.  The panel with the **Largest Overlap** is selected as "Panel A".
5.  We report the **Total Area** of Panel A (Inside + Outside the buffer).

***

##  License
This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments
*   **Satellite Imagery:** Esri World Imagery (Wayback)
*   **Datasets:** Roboflow Universe (Alfred Weber, LSGI547, Piscinas)
*   **Framework:** Ultralytics YOLOv8

***

## Judging / Quick Start

See `JUDGE.md` for concise reproduction steps for judges (venv setup, how to run `pipeline.py`, and expected outputs).
