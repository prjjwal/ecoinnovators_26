"""
EcoInnovators Ideathon 2026 - Solar Detection Pipeline
------------------------------------------------------
Team: 52_Saints
Model: YOLOv8l-OBB (Large)
Logic: Full Context Inference -> Geometric Intersection Filtering

Description:
1. Fetches satellite imagery at Zoom 18 (matches training resolution).
2. Stitches a 3x3 grid (768x768) and center-crops to 640x640 (matches model input).
3. Runs YOLOv8-OBB inference on the full context.
4. Filters results: A site is 'has_solar=True' ONLY if a panel geometrically intersects
   the 1200 sq.ft (or 2400 sq.ft) buffer zone.
"""

import os
import sys
import json
import cv2
import math
import argparse
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from shapely.geometry import Polygon, Point
from datetime import datetime

# --- CONFIGURATION ---
# Esri Wayback Tile Service (Free, Public XYZ Tiles)
ESRI_TILE_URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
HEADERS = {'User-Agent': 'EcoInnovators-Solar-Bot/2.0'}

# Set to 18 to match Training Data resolution (~0.6m/px)
ZOOM_LEVEL = 18
# Esri Native Tile Size
TILE_SIZE = 256
# Model Input Size (Center Crop Target)
MODEL_INPUT_SZ = 640

# Inference Thresholds
CONF_THRESHOLD = 0.35  # Optimal threshold for maximum F1
QC_BLUR_THRESHOLD = 50.0 # Laplacian variance threshold for blur rejection
SQFT_TO_SQM = 0.092903 # Standard conversion factor


# Set to True to save images of every site (with/without detections) to 'output_reults/debug_views/' post inference
DEBUG_MODE = True


class SolarPipeline:
    def __init__(self, model_path, output_dir):
        self.output_dir = output_dir
        self.artifacts_dir = os.path.join(output_dir, 'artifacts')
        self.debug_dir = os.path.join(output_dir, 'debug_views')
        
        os.makedirs(self.artifacts_dir, exist_ok=True)
        if DEBUG_MODE:
            os.makedirs(self.debug_dir, exist_ok=True)
        
        print(f"[INFO] Loading OBB model from {model_path}...")
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            print(f"[ERROR] Model file {model_path} not found. Exiting.")
            sys.exit(1)

    # --- FETCHER LOGIC ---
    def lat_lon_to_tile(self, lat, lon, zoom):
        """Converts Lat/Lon to Web Mercator XYZ Tile coordinates."""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return xtile, ytile

    def fetch_single_tile(self, z, x, y):
        """Fetches a single 256x256 tile from Esri."""
        url = ESRI_TILE_URL.format(z=z, x=x, y=y)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=3)
            if resp.status_code == 200:
                arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                return cv2.imdecode(arr, -1)
        except: pass
        return None

    def fetch_stitched_grid(self, lat, lon):
        """Fetches 3x3 grid (768x768 pixels) centered on lat/lon."""
        cx, cy = self.lat_lon_to_tile(lat, lon, ZOOM_LEVEL)
        canvas = np.zeros((TILE_SIZE * 3, TILE_SIZE * 3, 3), dtype=np.uint8)
        
        # Parallel Fetch for speed
        tasks = []
        with ThreadPoolExecutor(max_workers=9) as executor:
            for i, dx in enumerate([-1, 0, 1]):
                for j, dy in enumerate([-1, 0, 1]):
                    tasks.append(executor.submit(self.fetch_single_tile, ZOOM_LEVEL, cx + dx, cy + dy))
        
        results = [t.result() for t in tasks]
        
        # Stitch
        idx = 0; valid = 0
        for i in range(3):
            for j in range(3):
                tile = results[idx]; idx+=1
                if tile is not None:
                    y_s, x_s = j * TILE_SIZE, i * TILE_SIZE
                    canvas[y_s:y_s+TILE_SIZE, x_s:x_s+TILE_SIZE] = tile
                    valid += 1
        
        if valid < 5: return None # Fail if mostly empty (ocean/error)
        return canvas

    def check_qc(self, image):
        """Quality Control: Checks for Blur (Laplacian) and Contrast."""
        if image is None: return 0.0, False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blur Check
        if cv2.Laplacian(gray, cv2.CV_64F).var() < QC_BLUR_THRESHOLD: 
            return 0.0, False
        
        # Contrast Check (reject flat gray/white images)
        if np.std(gray) < 10.0: 
            return 0.0, False
            
        return 1.0, True

    def fetch_best_imagery(self, lat, lon):
        """
        Jitter Strategy: Fetches Center + 4 neighbors.
        Returns the sharpest image (Highest edge variance).
        """
        jitter = 0.000025 # ~2.5 meters at Equator
        offsets = [(0,0), (jitter,0), (0,jitter), (-jitter,0), (0,-jitter)]
        best_img = None; best_score = -1.0
        
        for dlat, dlon in offsets:
            img = self.fetch_stitched_grid(lat + dlat, lon + dlon)
            if img is not None and np.mean(img) > 10: # Basic check
                # Calculate sharpness
                score = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                if score > best_score:
                    best_score = score; best_img = img

        if best_img is None:
            # Return black image if all fail (prevents crash)
            best_img = np.zeros((TILE_SIZE*3, TILE_SIZE*3, 3), dtype=np.uint8)
            
        meta = {
            "source": "Esri World Imagery", 
            "capture_date": datetime.now().strftime("%Y-%m-%d"), 
            "zoom_level": ZOOM_LEVEL
        }
        return best_img, meta

    # --- PIPELINE LOGIC ---
    def calculate_gsd(self, lat):
        """Ground Sample Distance (meters/pixel) for current Zoom."""
        return (156543.03 * math.cos(math.radians(lat))) / (2 ** ZOOM_LEVEL)

    def save_debug_view(self, sid, image, gsd, detections=None):
        """Saves image with search radii (Red/Blue) and Raw Detections (Cyan)."""
        debug_img = image.copy()
        cx, cy = image.shape[1]//2, image.shape[0]//2
        
        # Draw Search Radii
        r1200 = int(math.sqrt((1200*SQFT_TO_SQM)/math.pi) / gsd)
        cv2.circle(debug_img, (cx, cy), r1200, (0, 0, 255), 2) # Red = 1200 sqft
        
        r2400 = int(math.sqrt((2400*SQFT_TO_SQM)/math.pi) / gsd)
        cv2.circle(debug_img, (cx, cy), r2400, (255, 0, 0), 2) # Blue = 2400 sqft
        
        # Draw RAW Detections (Cyan)
        if detections and detections.obb is not None:
            for obb in detections.obb:
                corners = obb.xyxyxyxy[0].cpu().numpy().astype(int)
                cv2.polylines(debug_img, [corners], True, (255, 255, 0), 1)

        cv2.imwrite(os.path.join(self.debug_dir, f"debug_{sid}.jpg"), debug_img)

    def process_file(self, input_path):
        if not os.path.exists(input_path): return
        df = pd.read_excel(input_path)
        final_results = []
        print(f"[INFO] Processing {len(df)} sites...")

        for idx, row in df.iterrows():
            row = {k.lower(): v for k, v in row.items()}
            sid = row.get('sample_id')
            lat = row.get('latitude') or row.get('lat')
            lon = row.get('longitude') or row.get('lon')
            if sid is None: continue

            # 1. Fetch High-Res Image (768x768)
            raw_image, meta = self.fetch_best_imagery(lat, lon)
            gsd = self.calculate_gsd(lat)

            # 2. Center Crop to 640x640 (Model Native Size)
            # This ensures 1:1 pixel scale (no resizing) and matches training data
            h, w = raw_image.shape[:2]
            start_y = (h - MODEL_INPUT_SZ) // 2
            start_x = (w - MODEL_INPUT_SZ) // 2
            inference_image = raw_image[start_y:start_y+MODEL_INPUT_SZ, start_x:start_x+MODEL_INPUT_SZ]

            # 3. Run Inference on FULL Context
            results = self.model.predict(inference_image, conf=CONF_THRESHOLD, verbose=False)
            all_detections = results[0] if len(results) > 0 else None
            
            # 4. Debug Dump
            if DEBUG_MODE:
                self.save_debug_view(sid, inference_image, gsd, all_detections)

            # 5. QC Check
            qc_score, is_valid = self.check_qc(inference_image)
            qc_status = "VERIFIABLE" if is_valid else "NOT_VERIFIABLE"

            # Init Record
            record = {
                "sample_id": int(sid), "lat": float(lat), "lon": float(lon),
                "has_solar": False, "confidence": 0.0, "pv_area_sqm_est": 0.0,
                "buffer_radius_sqft": 1200, "qc_status": qc_status,
                "bbox_or_mask": "", "image_metadata": meta
            }

            # 6. Geometric Filtering Logic
            if qc_status == "VERIFIABLE" and all_detections and all_detections.obb is not None:
                
                # Center of the crop is (320, 320)
                cx, cy = MODEL_INPUT_SZ // 2, MODEL_INPUT_SZ // 2
                
                # Calculate Radii in Pixels
                r1200 = int(math.sqrt((1200*SQFT_TO_SQM)/math.pi) / gsd)
                r2400 = int(math.sqrt((2400*SQFT_TO_SQM)/math.pi) / gsd)
                
                # Create Shapely Buffers
                poly_1200 = Point(cx, cy).buffer(r1200)
                poly_2400 = Point(cx, cy).buffer(r2400)
                
                # Variables to select Best Panel
                best_panel_area = 0.0
                best_overlap_area = 0.0
                best_mask_points = []
                max_conf = 0.0
                final_buffer_viz_r = r1200 # Default viz
                
                # --- Pass 1: Check 1200 sq.ft Buffer ---
                panels_in_1200 = []
                for obb in all_detections.obb:
                    corners = obb.xyxyxyxy[0].cpu().numpy().astype(int)
                    panel_poly = Polygon(corners)
                    
                    if panel_poly.intersects(poly_1200):
                        overlap = panel_poly.intersection(poly_1200).area
                        panels_in_1200.append({
                            'poly': panel_poly, 'corners': corners, 'overlap': overlap, 'conf': float(obb.conf)
                        })

                # Decide if 1200 matches
                if len(panels_in_1200) > 0:
                    record["buffer_radius_sqft"] = 1200
                    # Select best by Max Overlap (FAQ #4 Rule)
                    best = max(panels_in_1200, key=lambda x: x['overlap'])
                    best_overlap_area = best['overlap']
                    best_panel_area = best['poly'].area * (gsd**2) # Report TOTAL Area
                    best_mask_points = best['corners'].tolist()
                    max_conf = best['conf']
                    final_buffer_viz_r = r1200
                else:
                    # --- Pass 2: Check 2400 sq.ft Buffer ---
                    panels_in_2400 = []
                    for obb in all_detections.obb:
                        corners = obb.xyxyxyxy[0].cpu().numpy().astype(int)
                        panel_poly = Polygon(corners)
                        
                        if panel_poly.intersects(poly_2400):
                            overlap = panel_poly.intersection(poly_2400).area
                            panels_in_2400.append({
                                'poly': panel_poly, 'corners': corners, 'overlap': overlap, 'conf': float(obb.conf)
                            })
                    
                    if len(panels_in_2400) > 0:
                        record["buffer_radius_sqft"] = 2400
                        best = max(panels_in_2400, key=lambda x: x['overlap'])
                        best_overlap_area = best['overlap']
                        best_panel_area = best['poly'].area * (gsd**2)
                        best_mask_points = best['corners'].tolist()
                        max_conf = best['conf']
                        final_buffer_viz_r = r2400

                # --- Final Decision ---
                if best_overlap_area > 0:
                    record["has_solar"] = True
                    record["confidence"] = round(max_conf, 2)
                    record["pv_area_sqm_est"] = round(best_panel_area, 2)
                    record["bbox_or_mask"] = json.dumps([best_mask_points])
                    
                    # Generate Audit Artifact
                    audit_image = inference_image.copy()
                    
                    # Draw Buffer (Yellow)
                    cv2.circle(audit_image, (cx, cy), final_buffer_viz_r, (0, 255, 255), 2)
                    
                    # Draw Chosen Panel (Green)
                    pts = np.array(best_mask_points, dtype=np.int32)
                    cv2.polylines(audit_image, [pts], True, (0, 255, 0), 3)
                    
                    cv2.imwrite(os.path.join(self.artifacts_dir, f"{sid}_audit.jpg"), audit_image)

            final_results.append(record)
            status = "FOUND" if record["has_solar"] else "NONE"
            print(f"[{idx+1}/{len(df)}] ID:{sid} | {status} | Area:{record['pv_area_sqm_est']}m2")

        with open(os.path.join(self.output_dir, "predictions.json"), "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"\n[SUCCESS] Results saved to {self.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="input_data")
    parser.add_argument("--output_dir", type=str, default="output_results")
    parser.add_argument("--model", type=str, default="solar_model.pt")
    args = parser.parse_args()
    
    input_xlsx = os.path.join(args.input_dir, "input.xlsx")
    pipeline = SolarPipeline(model_path=args.model, output_dir=args.output_dir)
    pipeline.process_file(input_xlsx)
