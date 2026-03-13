"""
Flask backend API for Treelance workflow management.
Treelance - Tree Vitality Analysis by LiveEO
Provides REST endpoints for workflow execution, status tracking, and result visualization.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import subprocess
import threading
import os
import sys
import json
import time
import re
import shlex
from pathlib import Path
from datetime import datetime
from typing import Literal, get_args
import yaml
import logging
import tempfile
import base64
import io
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
import numpy as np

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import OperationalError
except ImportError:
    create_engine = None
    text = None
    OperationalError = Exception

# Set up logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Add parent directory to Python path to find treelance_sentinel module
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# =============================================================================
# Security Configuration
# =============================================================================

# Allowed CORS origins (configure via environment variable for production)
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:5000,http://127.0.0.1:5000').split(',')

# Valid step IDs for input validation
VALID_NORMAL_STEPS = Literal[
    'normal_download', 'normal_asset_prep', 'normal_process_imagery',
    'normal_segmentation', 'normal_prediction', 'normal_folium'
]
VALID_TIME_SERIES_STEPS = Literal[
    'download', 'preprocess', 'segmentation', 'prediction',
    'change_detection', 'tree_clustering', 'vitality', 'all'
]
VALID_WORKFLOWS = Literal['normal', 'time_series']

# Debug mode - NEVER enable in production
DEBUG_MODE = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

app = Flask(__name__, static_folder='static', static_url_path='')

# Configure CORS with restricted origins
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True, methods=['GET', 'POST'])


# =============================================================================
# Security Middleware
# =============================================================================

@app.after_request
def set_security_headers(response):
    """Add security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # Content Security Policy - allow Leaflet/Leaflet.draw from unpkg for map
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://unpkg.com https://fonts.googleapis.com; "
        "img-src 'self' data: https: blob:; "
        "font-src 'self' https://fonts.gstatic.com; "
        "connect-src 'self' https://*.tile.openstreetmap.org;"
    )
    return response


def sanitize_path_component(value: str) -> str:
    """Remove dangerous characters from path components to prevent path traversal.
    
    Args:
        value: The path component to sanitize
        
    Returns:
        Sanitized string safe for use in paths
        
    Raises:
        ValueError: If the value is invalid or potentially malicious
    """
    if not value or not isinstance(value, str):
        raise ValueError("Path component cannot be empty")
    
    # Remove any path traversal attempts
    sanitized = value.replace('..', '').replace('/', '_').replace('\\', '_')
    
    # Only allow alphanumeric, underscore, hyphen, and dot
    sanitized = re.sub(r'[^a-zA-Z0-9_\-.]', '_', sanitized)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    if not sanitized or sanitized in ('.', '..'):
        raise ValueError(f"Invalid path component: {value}")
    
    return sanitized


def validate_step_id(step_id: str, workflow: str) -> bool:
    """Validate that a step ID is allowed for the given workflow.
    
    Args:
        step_id: The step ID to validate
        workflow: The workflow type ('normal' or 'time_series')
        
    Returns:
        True if valid, False otherwise
    """
    if workflow == 'normal':
        valid_steps = ['normal_download', 'normal_asset_prep', 'normal_process_imagery',
                       'normal_segmentation', 'normal_prediction', 'normal_folium']
    else:
        valid_steps = ['download', 'preprocess', 'segmentation', 'prediction',
                       'change_detection', 'tree_clustering', 'vitality', 'all']
    
    return step_id in valid_steps


def validate_workflow(workflow: str) -> bool:
    """Validate that a workflow type is allowed.
    
    Args:
        workflow: The workflow type to validate
        
    Returns:
        True if valid, False otherwise
    """
    return workflow in ['normal', 'time_series']


def validate_date_format(date_str: str) -> bool:
    """Validate that a date string is in ISO format (YYYY-MM-DD).
    
    Args:
        date_str: The date string to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not date_str:
        return True  # Empty dates are allowed (optional)
    
    try:
        datetime.fromisoformat(date_str)
        return True
    except (ValueError, TypeError):
        return False

# Workflow state management
workflow_state = {
    "current_run": None,
    "status": "idle",  # idle, running, completed, error
    "current_step": None,
    "steps": {},
    "logs": [],
    "results": {},
    "last_config": {}  # Store last config for dashboard
}

# Workflow steps configuration
# We support two workflows:
# - "normal": single-date classification (vegbins mode)
# - "time_series": two-date comparison + vitality (time_series mode)
WORKFLOW_STEPS = {
    "normal": [
        {
            "id": "normal_download",
            "name": "Download Imagery",
            "description": "Download Sentinel-2 imagery for a single date",
            "details": "Searches Sentinel-2 catalog for the AOI and date, filters scenes by cloud cover, and downloads the raw bands listed in the config.",
            "icon": "📥",
        },
        {
            "id": "normal_asset_prep",
            "name": "Asset Preparation",
            "description": "Prepare AOI assets and tiles",
            "details": "Normalizes and validates the AOI geometry, generates internal tile indexes, and sets up the directory structure for downstream processing.",
            "icon": "🧩",
        },
        {
            "id": "normal_process_imagery",
            "name": "Process Imagery",
            "description": "Extract, merge and tile bands",
            "details": "Reads the downloaded Sentinel-2 bands, reprojects and resamples them, stacks them into analysis-ready rasters, and cuts them into tiles.",
            "icon": "⚙️",
        },
        {
            "id": "normal_segmentation",
            "name": "Segmentation",
            "description": "K-means segmentation over imagery",
            "details": "Computes vegetation indices and runs k-means clustering to segment the scene into homogeneous vegetation patches.",
            "icon": "🔲",
        },
        {
            "id": "normal_prediction",
            "name": "Prediction",
            "description": "Tree classification with ML model",
            "details": "Applies the trained tree model to each segment/tile to predict vegetation classes and outputs classification rasters and summaries.",
            "icon": "🌳",
        },
        {
            "id": "normal_folium",
            "name": "Folium Map",
            "description": "Create interactive Folium visualization",
            "details": "Builds an interactive Folium web map that overlays predictions and key layers for quick visual QA and exploration.",
            "icon": "🗺️",
        },
    ],
    "time_series": [
        {
            "id": "download",
            "name": "Download Imagery",
            "description": "Download Sentinel-2 imagery based on temporal mode selection",
            "details": "Downloads Sentinel-2 snapshots for change analysis. Number of time steps depends on temporal mode: Bi-temporal (2 dates) or Tri-temporal (3 dates with pre-greenup baseline).",
            "insights": [
                "Queries AWS Earth Search API for Sentinel-2 L2A satellite imagery",
                "Filters scenes by cloud cover (< 10% default) for best quality",
                "Downloads 8 spectral bands (B02, B03, B04, B05, B06, B08, B11, B12)",
                "Selects best scene per MGRS tile based on cloud cover or date",
                "Bi-temporal mode: Downloads Peak Vitality (T1, mid-summer) + Current Analysis (T2, detection date)",
                "Tri-temporal mode: Downloads 3 time steps:",
                "  • T0 (Pre-Greenup): Early spring baseline - captures soil/bark before leaves hide everything",
                "  • T1 (Peak Vitality): Mid-summer reference - maximum chlorophyll for healthy baseline",
                "  • T2 (Current Analysis): Your actual detection date",
                "Tri-temporal enables phenological normalization: distinguishes drought from seasonal changes"
            ],
            "icon": "📥",
        },
        {
            "id": "preprocess",
            "name": "Preprocess Imagery",
            "description": "Stack bands and apply cloud masking",
            "details": "Builds cloud-masked, co-registered band stacks for both time steps, ready for segmentation and model inference.",
            "insights": [
                "Stacks all spectral bands into a single multi-band GeoTIFF",
                "Applies cloud masking using SCL (Scene Classification Layer)",
                "Calculates vegetation indices: NDVI and NDMI",
                "Ensures all bands are pixel-aligned and co-registered",
                "Prepares data for segmentation and classification"
            ],
            "icon": "⚙️",
        },
        {
            "id": "segmentation",
            "name": "Segmentation",
            "description": "K-means clustering and polygonization",
            "details": "Runs NDVI-based k-means segmentation on the current snapshot and converts clusters into polygons for zonal statistics and training.",
            "insights": [
                "Uses NDVI-split K-means: separates high NDVI (trees) from low NDVI (urban/grass)",
                "High NDVI areas: 30 clusters (captures tree diversity)",
                "Low NDVI areas: 5 clusters (urban/grassland regions)",
                "Converts classified raster to polygons for analysis",
                "Calculates zonal statistics (mean band values) per polygon",
                "Adds H3 geo-embeddings for spatial context"
            ],
            "icon": "🔲",
        },
        {
            "id": "prediction",
            "name": "Tree Prediction",
            "description": "ML model prediction and classification",
            "details": "Uses the tree classification model to assign species/risk classes to each pixel/tile for both time steps. Fine-tuning uses labels from manually labeled confidence GPKGs (see insights).",
            "insights": [
                "Applies Deep U-Net classifier with attention mechanisms",
                "Classifies each polygon into: Tree, Grassland, or Urban",
                "Calculates confidence scores and writes low-confidence polygons to predictions/confidence/ (for labeling)",
                "Fine-tuning: labels do NOT come from this step automatically. Run the flow once to generate confidence GPKGs, then manually set class_id (0=grassland, 1=tree, 2=urban) in those GPKGs (e.g. in QGIS) under the same local AOI output path. When you run Tree Prediction again (or Run All), the pipeline will find labeled confidence files and fine-tune the model.",
                "For rectangle AOI: outputs go to local_output/custom_rectangle_<ts>/time_series/<snapshot>/predictions/confidence/; label those GPKGs and re-run Tree Prediction to fine-tune."
            ],
            "icon": "🌳",
        },
        {
            "id": "change_detection",
            "name": "Change Detection",
            "description": "Organize time series for change detection",
            "details": "Aligns predictions and imagery from snapshots into a common layout, computing NDVI/NDMI deltas per tile. Supports tri-temporal mode with pre-greenup baseline for phenological normalization.",
            "insights": [
                "Rasterizes tree predictions to create tree mask (with fallback to tree_clusters)",
                "Clips time steps to tree-covered areas only",
                "Bi-temporal: Exports time_step_1.tif (T1: Peak Vitality) and time_step_2.tif (T2: Current Analysis)",
                "Tri-temporal: Exports time_step_0.tif (T0: Pre-Greenup), time_step_1.tif (T1: Peak Vitality), time_step_2.tif (T2: Current Analysis)",
                "Includes NDVI and NDMI bands for all time periods",
                "Prepares data for pixel-level vitality analysis with phenological context",
                "Ensures consistent tile coverage across time steps with automatic reprojection"
            ],
            "icon": "📊",
        },
        {
            "id": "tree_clustering",
            "name": "Tree Clustering",
            "description": "HDBSCAN clustering for tree species",
            "details": "Builds species-like clusters in feature space (B5, B6, B8, B11, NDVI, ΔNDVI) using PCA + HDBSCAN to separate tree communities.",
            "insights": [
                "Uses red-edge bands (B05, B06) for species discrimination",
                "Applies PCA dimensionality reduction (removes noise)",
                "Clusters trees into species-like groups using HDBSCAN",
                "Each cluster represents a tree community/species",
                "Enables 'Mixture of Experts' approach for vitality",
                "Outputs: Cluster raster with unique IDs per tree group"
            ],
            "icon": "🎯",
        },
        {
            "id": "vitality",
            "name": "Vitality Analysis (Treelance 2.0)",
            "description": "Advanced autoencoder-based vitality classification with Treelance 2.0 features",
            "details": "Trains cluster-specific autoencoders on healthy pixels and scores deviations to label tree vitality/morbidity. Supports bi-temporal and tri-temporal modes with advanced architectures. Tri-temporal enables phenological normalization to distinguish drought from seasonal changes.",
            "insights": [
                "Architecture: Conv-Transformer (baseline) or U-ConvTransformer (2.0) with dual-branch attention",
                "Bi-temporal: Uses 4 features [prev_NDVI, prev_NDMI, curr_NDVI, curr_NDMI] from T1 (Peak Vitality) + T2 (Current)",
                "Tri-temporal: Uses 6 features [prev_prev_NDVI, prev_prev_NDMI, prev_NDVI, prev_NDMI, curr_NDVI, curr_NDMI]",
                "Tri-temporal benefit: If T1 was high and T2 is low, is it drought or early autumn? T0 (pre-greenup) lets the model normalize for seasonal timing",
                "Spectral Attention: Prioritizes B5 (Red Edge) and B11 (SWIR) bands for early morbidity detection",
                "Weighted Loss: Emphasizes critical moisture bands (B5=2.5x, B11=3.0x weight)",
                "Deep Feature Differencing: Combines reconstruction error + latent space distance",
                "Twin-Input Reconstruction: Siamese task (T1+T2 → reconstruct T1 baseline)",
                "U-Net Skip Connections: Prevents blurring of small tree crowns",
                "Classifies into 5 classes: High/Medium Vitality, Medium/Severe Morbidity, No Change",
                "Outputs: Pixel-level vitality classification raster with colormap"
            ],
            "icon": "💚",
        },
    ],
}


def _categorize_log_message(msg: str) -> str:
    """Assign a category so logs can be filtered: error, warning, success, progress, info."""
    if not msg or not isinstance(msg, str):
        return "info"
    m = msg.lower()
    if any(x in m for x in ("error", "failed", "exception", "traceback", "exit code", "critical")):
        return "error"
    if any(x in m for x in ("warning", "warn", "skip", "insufficient", "could not", "fallback", "not found")):
        return "warning"
    if any(x in m for x in ("completed", "saved", "uploaded", "finished", "success", "✅", "done", "complete")):
        return "success"
    if any(x in m for x in ("processing", "downloading", "running", "starting", "step", "tile_", "band ", "polygons", "clusters", "predictions", "segmentation")):
        return "progress"
    return "info"


def _simplify_log_message(msg: str, max_len: int = 400) -> str:
    """Shorten paths and noisy prefixes so each line is informative."""
    if not msg or not isinstance(msg, str):
        return msg or ""
    import re
    s = msg.strip()
    # Strip loguru-style prefix: "2025-01-28 12:00:00 | INFO    | message"
    s = re.sub(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*\|\s*\w+\s*\|\s*", "", s)
    # Shorten S3 paths: s3://bucket/long/prefix/.../file -> …/file
    s = re.sub(r"s3://[^\s/]+(?:/[^\s/]+)*/([^\s/]+)(?=\s|$|[,.)])", r"…/\1", s)
    # Shorten long local paths: /path/to/very/long/dir/file -> …/file (keep preceding space/paren)
    s = re.sub(r"(^|[\s(])(/[^\s/]+(?:/[^\s/]+){3,}/)([^\s/]+)(?=\s|$|[,.)])", r"\1…/\3", s)
    s = s.strip()
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def get_workflow_status():
    """Get current workflow status. Logs are enriched with category and simplified message for UI."""
    raw_logs = workflow_state["logs"][-200:]
    enriched = []
    for entry in raw_logs:
        if not isinstance(entry, dict):
            continue
        msg = entry.get("message", "")
        category = _categorize_log_message(msg)
        display = _simplify_log_message(msg)
        enriched.append({
            "timestamp": entry.get("timestamp"),
            "step": entry.get("step"),
            "message": msg,
            "category": category,
            "message_display": display or msg,
        })
    return {
        "status": workflow_state["status"],
        "current_step": workflow_state["current_step"],
        "steps": workflow_state["steps"],
        "logs": enriched,
        "results": workflow_state["results"]
    }


@app.route('/api/status', methods=['GET'])
def status():
    """Get workflow status."""
    return jsonify(get_workflow_status())


@app.route('/api/steps', methods=['GET'])
def get_steps():
    """Get available workflow steps."""
    return jsonify(WORKFLOW_STEPS)


@app.route('/api/run-step', methods=['POST'])
def run_step():
    """Run a specific workflow step.
    
    Security: Validates all inputs before processing.
    """
    data = request.json
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    step_id = data.get('step_id')
    config = data.get('config', {})
    workflow = data.get('workflow', 'time_series')
    
    # Security: Validate workflow type
    if not validate_workflow(workflow):
        logger.warning(f"Invalid workflow type attempted: {workflow}")
        return jsonify({"error": f"Invalid workflow type: {workflow}"}), 400
    
    # Security: Validate step_id
    if not step_id or not validate_step_id(step_id, workflow):
        logger.warning(f"Invalid step_id attempted: {step_id} for workflow {workflow}")
        return jsonify({"error": f"Invalid step_id: {step_id}"}), 400
    
    # Security: Validate config is a dictionary
    if not isinstance(config, dict):
        return jsonify({"error": "Config must be a dictionary"}), 400
    
    # Security: Validate date format if provided
    if config.get('date') and not validate_date_format(config.get('date')):
        return jsonify({"error": f"Invalid date format: {config.get('date')}"}), 400
    
    if workflow_state["status"] == "running":
        return jsonify({"error": "Workflow is already running"}), 400
    
    # Find step configuration within selected workflow
    steps_for_workflow = WORKFLOW_STEPS.get(workflow, [])
    step_config = next((s for s in steps_for_workflow if s["id"] == step_id), None)
    if not step_config:
        return jsonify({"error": f"Unknown step: {step_id} for workflow {workflow}"}), 400
    
    # Start workflow in background thread
    thread = threading.Thread(
        target=execute_workflow_step,
        args=(step_id, step_config, config, workflow),
        daemon=True
    )
    thread.start()
    
    return jsonify({"message": f"Started step: {step_id}", "status": "running"})


@app.route('/api/run-all', methods=['POST'])
def run_all():
    """Run all workflow steps sequentially.
    
    Security: Validates all inputs before processing.
    """
    data = request.json
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    config = data.get('config', {})
    workflow = data.get('workflow', 'time_series')
    
    # Security: Validate workflow type
    if not validate_workflow(workflow):
        logger.warning(f"Invalid workflow type attempted: {workflow}")
        return jsonify({"error": f"Invalid workflow type: {workflow}"}), 400
    
    # Security: Validate config is a dictionary
    if not isinstance(config, dict):
        return jsonify({"error": "Config must be a dictionary"}), 400
    
    # Security: Validate date format if provided
    if config.get('date') and not validate_date_format(config.get('date')):
        return jsonify({"error": f"Invalid date format: {config.get('date')}"}), 400
    
    if workflow_state["status"] == "running":
        return jsonify({"error": "Workflow is already running"}), 400
    
    # Start full workflow in background thread
    thread = threading.Thread(
        target=execute_full_workflow,
        args=(config, workflow),
        daemon=True
    )
    thread.start()
    
    return jsonify({"message": "Started full workflow", "status": "running"})


def _bbox_to_geojson(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> dict:
    """Build a GeoJSON FeatureCollection for a rectangular AOI (WGS84)."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [min_lon, min_lat],
                            [max_lon, min_lat],
                            [max_lon, max_lat],
                            [min_lon, max_lat],
                            [min_lon, min_lat],
                        ]
                    ],
                },
            }
        ],
    }


@app.route('/api/run-rectangle', methods=['POST'])
def run_rectangle():
    """Run full time-series workflow on a rectangle AOI drawn on the map.
    
    Body: { "bbox": { "minLon", "minLat", "maxLon", "maxLat" }, "config": { ... } }
    Writes a temporary GeoJSON, sets config.aoi to that path and config.aoi_suffix for S3 output, then runs execute_full_workflow.
    """
    data = request.json
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    bbox = data.get("bbox")
    if not bbox or not isinstance(bbox, dict):
        return jsonify({"error": "Missing or invalid 'bbox' (minLon, minLat, maxLon, maxLat)"}), 400

    try:
        min_lon = float(bbox.get("minLon"))
        min_lat = float(bbox.get("minLat"))
        max_lon = float(bbox.get("maxLon"))
        max_lat = float(bbox.get("maxLat"))
    except (TypeError, ValueError):
        return jsonify({"error": "bbox values must be numbers"}), 400

    if min_lon >= max_lon or min_lat >= max_lat:
        return jsonify({"error": "Invalid bbox: minLon < maxLon and minLat < maxLat"}), 400

    if workflow_state["status"] == "running":
        return jsonify({"error": "Workflow is already running"}), 400

    project_root = Path(__file__).parent.parent
    tmp_aoi_dir = project_root / "tmp_aoi"
    tmp_aoi_dir.mkdir(exist_ok=True)

    ts = int(time.time())
    aoi_filename = f"aoi_rectangle_{ts}.geojson"
    aoi_path = tmp_aoi_dir / aoi_filename
    geojson = _bbox_to_geojson(min_lon, min_lat, max_lon, max_lat)
    with open(aoi_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f)

    config = data.get("config", {})
    if not isinstance(config, dict):
        config = {}
    config = dict(config)
    config["aoi"] = str(aoi_path)
    config["aoi_suffix"] = f"custom_rectangle_{ts}"
    if not config.get("date"):
        config["date"] = data.get("timeSeriesDate") or "2025-06-01"
    if data.get("overwrite_cache") is not None:
        config["overwrite_cache"] = data["overwrite_cache"]

    workflow = data.get("workflow", "time_series")
    if not validate_workflow(workflow):
        return jsonify({"error": f"Invalid workflow: {workflow}"}), 400
    if config.get("date") and not validate_date_format(config.get("date")):
        return jsonify({"error": f"Invalid date: {config.get('date')}"}), 400

    thread = threading.Thread(
        target=execute_full_workflow,
        args=(config, workflow),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "message": "Started full workflow on rectangle AOI",
        "status": "running",
        "aoi_path": str(aoi_path),
        "aoi_suffix": config["aoi_suffix"],
    })


# =============================================================================
# Get AOI geometry from database (for map display and area/bounds)
# =============================================================================

@app.route("/api/aoi", methods=["GET"])
def get_aoi():
    """Load AOI geometry from database (or S3/file) and return GeoJSON plus area and bounds.
    Query: ?aoi=<schema.table or schema.table.name or file path or S3 path>
    Returns: { "geojson": FeatureCollection, "area_km2": float, "bounds": [minLon, minLat, maxLon, maxLat] }"""
    aoi = request.args.get("aoi", "").strip()
    if not aoi:
        return jsonify({"error": "Missing 'aoi' query parameter"}), 400
    try:
        from treelance_sentinel.sentinel_download.sentinel_aws_downloader import _load_geometry

        geom = _load_geometry(aoi)
        geom_gdf = gpd.GeoDataFrame.from_features([
            {"type": "Feature", "properties": {}, "geometry": geom}
        ], crs="EPSG:4326")

        # Area in km² (project to Web Mercator for meters)
        geom_projected = geom_gdf.to_crs(epsg=3857)
        area_sqm = float(geom_projected.geometry.area.sum())
        area_km2 = area_sqm / 1_000_000

        # Bounds [minLon, minLat, maxLon, maxLat]
        minx, miny, maxx, maxy = geom_gdf.total_bounds

        geojson = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": geom}],
        }
        return jsonify({
            "geojson": geojson,
            "area_km2": round(area_km2, 2),
            "bounds": [float(minx), float(miny), float(maxx), float(maxy)],
        })
    except Exception as e:
        logger.warning("Failed to load AOI for display: %s", e)
        return jsonify({"error": str(e)}), 404


# =============================================================================
# Upload drawn AOI to database (same DB as prediction_ready_aoi)
# =============================================================================

DRAWNAOIS_SCHEMA = "prediction_ready_aoi"
DRAWNAOIS_TABLE = "drawn_aois"
DRAWNAOIS_FULL_TABLE = f'"{DRAWNAOIS_SCHEMA}"."{DRAWNAOIS_TABLE}"'


def _get_writable_db_url() -> str | None:
    """Return PostgreSQL URL with write access, or None if not configured."""
    url = os.environ.get("TREELANCE_DB_WRITE_URL", "").strip()
    if url:
        return url
    return None


@app.route("/api/upload-aoi", methods=["POST"])
def upload_aoi():
    """Persist a drawn AOI (bbox or GeoJSON) to the database table prediction_ready_aoi.drawn_aois.
    Body: { "name": "my_aoi", "bbox": { minLon, minLat, maxLon, maxLat } } or { "name", "geojson": {...} }
    Returns the AOI identifier to use in workflow: prediction_ready_aoi.drawn_aois.<name>"""
    data = request.json
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Missing or empty 'name'"}), 400
    name = sanitize_path_component(name)
    if not name:
        return jsonify({"error": "Invalid 'name' (use alphanumeric, underscore, hyphen)"}), 400

    geojson_geom: dict
    if "geojson" in data and data["geojson"]:
        gj = data["geojson"]
        if isinstance(gj, dict):
            if gj.get("type") == "FeatureCollection" and gj.get("features"):
                geojson_geom = _bbox_to_geojson(
                    *_bbox_from_feature_collection(gj)
                )
                geojson_geom = geojson_geom["features"][0]["geometry"]
            elif gj.get("type") == "Feature":
                geojson_geom = gj.get("geometry") or gj
            else:
                geojson_geom = gj
        else:
            return jsonify({"error": "Invalid 'geojson' (must be object)"}), 400
    elif "bbox" in data and isinstance(data["bbox"], dict):
        bbox = data["bbox"]
        try:
            min_lon = float(bbox.get("minLon"))
            min_lat = float(bbox.get("minLat"))
            max_lon = float(bbox.get("maxLon"))
            max_lat = float(bbox.get("maxLat"))
        except (TypeError, ValueError):
            return jsonify({"error": "bbox must have numeric minLon, minLat, maxLon, maxLat"}), 400
        if min_lon >= max_lon or min_lat >= max_lat:
            return jsonify({"error": "Invalid bbox"}), 400
        geojson_geom = _bbox_to_geojson(min_lon, min_lat, max_lon, max_lat)["features"][0]["geometry"]
    else:
        return jsonify({"error": "Provide either 'bbox' or 'geojson'"}), 400

    db_url = _get_writable_db_url()
    if not db_url:
        return (
            jsonify({
                "error": "Upload to database is not configured. Set TREELANCE_DB_WRITE_URL to your local PostGIS connection string.",
            }),
            503,
        )

    if create_engine is None or text is None:
        return jsonify({"error": "sqlalchemy is required for upload-aoi"}), 503

    try:
        engine = create_engine(db_url)
        geom_str = json.dumps(geojson_geom)
        with engine.connect() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS \"{DRAWNAOIS_SCHEMA}\""))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {DRAWNAOIS_FULL_TABLE} (
                        id SERIAL PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        geom GEOMETRY(Geometry, 4326),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
            )
            conn.execute(
                text(
                    f"""
                    INSERT INTO {DRAWNAOIS_FULL_TABLE} (name, geom)
                    VALUES (:name, ST_SetSRID(ST_GeomFromGeoJSON(:geom), 4326))
                    ON CONFLICT (name) DO UPDATE SET geom = EXCLUDED.geom, created_at = NOW()
                    """
                ),
                {"name": name, "geom": geom_str},
            )
            conn.commit()
        aoi_identifier = f"{DRAWNAOIS_SCHEMA}.{DRAWNAOIS_TABLE}.{name}"
        return jsonify({
            "message": "AOI saved to database",
            "aoi_identifier": aoi_identifier,
            "name": name,
        })
    except Exception as e:
        logger.exception("Upload AOI failed")
        return jsonify({"error": str(e)}), 500


def _bbox_from_feature_collection(fc: dict) -> tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) from a GeoJSON FeatureCollection."""
    from shapely.geometry import shape
    from shapely.ops import unary_union

    features = fc.get("features") or []
    if not features:
        raise ValueError("Empty FeatureCollection")
    geoms = [shape(f["geometry"]) for f in features]
    unioned = unary_union(geoms)
    minx, miny, maxx, maxy = unioned.bounds
    return (float(minx), float(miny), float(maxx), float(maxy))


# =============================================================================
# Outputs on map (lightweight: raster PNG previews for classification and vitality)
# =============================================================================

LOCAL_OUTPUTS_ROOT = os.environ.get("TREELANCE_OUTPUTS_DIR", str(Path(__file__).parent.parent / "local_output"))
RASTER_PREVIEW_SIZE = 512  # px for classification and vitality raster previews (sharper map overlays)
VITALITY_COLORMAP = {  # class -> (R, G, B, A)
    0: (180, 50, 50, 220),   # severe_morbidity - red
    1: (230, 140, 60, 220),  # medium_morbidity - orange
    2: (255, 220, 100, 220), # medium_vitality - yellow
    3: (80, 180, 80, 220),   # high_vitality - green
    4: (160, 160, 160, 180), # no_change - gray
}
# Classification (prediction) raster: 0=Grassland, 1=Tree, 2=Urban; 255=nodata
CLASSIFICATION_COLORMAP = {
    0: (134, 239, 172, 200),   # grassland - light green
    1: (22, 101, 52, 220),     # tree - dark green
    2: (120, 113, 108, 200),   # urban - gray
    255: (0, 0, 0, 0),         # nodata - transparent
}


def _outputs_base_dir(aoi_suffix: str) -> str:
    """Return local base output dir for an AOI suffix (no trailing slash)."""
    try:
        safe = sanitize_path_component(aoi_suffix)
    except ValueError:
        safe = "unknown"
    return str(Path(LOCAL_OUTPUTS_ROOT) / safe)


@app.route("/api/outputs/list", methods=["GET"])
def outputs_list():
    """List available map outputs for an AOI: classification rasters and vitality rasters.
    Query: aoi_suffix (required).
    Returns: { classification: { available, snapshots, rasters: [ { file, url } ] }, vitality: { available, tiles: [ { tile_id, image_url } ] } }
    """
    aoi_suffix = request.args.get("aoi_suffix", "").strip()
    if not aoi_suffix:
        return jsonify({"error": "Missing aoi_suffix"}), 400
    try:
        sanitize_path_component(aoi_suffix)
    except ValueError:
        return jsonify({"error": "Invalid aoi_suffix"}), 400

    base = Path(_outputs_base_dir(aoi_suffix))

    out = {"classification": {"available": False, "snapshots": [], "rasters": []}, "vitality": {"available": False, "tiles": []}}

    # Classification rasters: time_series/<snapshot>/predictions/raster/multiclass/*_classification.tif
    for snapshot in ["current", "peak_vitality", "previous", "pre_greenup"]:
        multiclass_dir = base / "time_series" / snapshot / "predictions" / "raster" / "multiclass"
        if multiclass_dir.exists():
            for path in multiclass_dir.glob("*_classification.tif"):
                basename = path.name
                out["classification"]["available"] = True
                if snapshot not in out["classification"]["snapshots"]:
                    out["classification"]["snapshots"].append(snapshot)
                out["classification"]["rasters"].append({
                    "file": basename,
                    "url": f"/api/outputs/classification-raster?aoi_suffix={aoi_suffix}&snapshot={snapshot}&file={basename}",
                })
    # If no prediction rasters, try segmentation classified rasters: time_series/<snapshot>/segmentation/classified/*_classified.tif
    if not out["classification"]["rasters"]:
        for snapshot in ["current", "peak_vitality", "previous", "pre_greenup"]:
            seg_dir = base / "time_series" / snapshot / "segmentation" / "classified"
            if seg_dir.exists():
                for path in seg_dir.glob("*_classified.tif"):
                    basename = path.name
                    out["classification"]["available"] = True
                    if snapshot not in out["classification"]["snapshots"]:
                        out["classification"]["snapshots"].append(snapshot)
                    out["classification"]["rasters"].append({
                        "file": basename,
                        "url": f"/api/outputs/classification-raster?aoi_suffix={aoi_suffix}&snapshot={snapshot}&file={basename}&type=segmentation",
                    })

    # Vitality: vitality_autoencoder/tile_<ID>/vitality_classes.tif
    vitality_dir = base / "vitality_autoencoder"
    if vitality_dir.exists():
        for tile_dir in vitality_dir.glob("tile_*"):
            tif_path = tile_dir / "vitality_classes.tif"
            if tile_dir.is_dir() and tif_path.exists():
                tile_id = tile_dir.name.replace("tile_", "")
                out["vitality"]["tiles"].append({
                    "tile_id": tile_id,
                    "image_url": f"/api/outputs/vitality-tile/{tile_id}?aoi_suffix={aoi_suffix}",
                })
    out["vitality"]["available"] = len(out["vitality"]["tiles"]) > 0

    return jsonify(out)


def _raster_to_preview_png(data: np.ndarray, colormap: dict, size: int = RASTER_PREVIEW_SIZE) -> tuple[bytes, tuple[int, int]]:
    """Resample 2D class raster to small size and render RGBA PNG with colormap. Returns (png_bytes, (h, w))."""
    h, w = data.shape
    step_h, step_w = max(1, h // size), max(1, w // size)
    small = data[::step_h, ::step_w][:size, :size].astype(np.int32)
    if small.shape[0] < size or small.shape[1] < size:
        padded = np.zeros((size, size), dtype=np.int32)
        padded[: small.shape[0], : small.shape[1]] = small
        small = padded
    out_h, out_w = small.shape[0], small.shape[1]
    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    for cls, color in colormap.items():
        mask = small == cls
        rgba[mask, 0] = color[0]
        rgba[mask, 1] = color[1]
        rgba[mask, 2] = color[2]
        rgba[mask, 3] = color[3]
    # Unmapped values -> transparent
    mapped = np.zeros((out_h, out_w), dtype=bool)
    for cls in colormap:
        mapped |= small == cls
    rgba[~mapped, 3] = 0
    buf = io.BytesIO()
    try:
        from PIL import Image
        img = Image.fromarray(rgba)
        img.save(buf, format="PNG")
    except Exception:
        import struct
        import zlib
        raw = rgba.tobytes()
        def png_chunk(ctype, data):
            chunk = ctype + data
            return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        ihdr = struct.pack(">IIBBBBB", out_w, out_h, 8, 6, 0, 0, 0)
        idat = zlib.compress(raw, 9)
        buf.write(b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", idat) + png_chunk(b"IEND", b""))
    buf.seek(0)
    return buf.getvalue(), (out_h, out_w)


@app.route("/api/outputs/classification-raster", methods=["GET"])
def outputs_classification_raster():
    """Return bounds (WGS84) and a small PNG preview of a classification raster (tree/grass/urban or segmentation)."""
    aoi_suffix = request.args.get("aoi_suffix", "").strip()
    snapshot = request.args.get("snapshot", "current").strip() or "current"
    file_name = request.args.get("file", "").strip()
    raster_type = request.args.get("type", "prediction").strip() or "prediction"
    if not aoi_suffix or not file_name:
        return jsonify({"error": "Missing aoi_suffix or file"}), 400
    try:
        sanitize_path_component(aoi_suffix)
        sanitize_path_component(snapshot)
        sanitize_path_component(file_name)
    except ValueError:
        return jsonify({"error": "Invalid parameter"}), 400
    if ".." in file_name or "/" in file_name or "\\" in file_name:
        return jsonify({"error": "Invalid file name"}), 400

    base = Path(_outputs_base_dir(aoi_suffix))
    if raster_type == "segmentation":
        tif_path = base / "time_series" / snapshot / "segmentation" / "classified" / file_name
    else:
        tif_path = base / "time_series" / snapshot / "predictions" / "raster" / "multiclass" / file_name

    if not tif_path.exists():
        return jsonify({"error": "Classification raster not found"}), 404

    with rasterio.open(tif_path) as src:
        crs = src.crs
        bounds = src.bounds
        data = src.read(1)
        if crs and not crs.is_geographic:
            w, s, e, n = transform_bounds(crs, "EPSG:4326", *bounds)
        else:
            w, s, e, n = bounds.left, bounds.bottom, bounds.right, bounds.top
        bounds_wgs84 = [[s, w], [n, e]]
        if raster_type == "segmentation":
            # Many classes: simple cyclic colormap (grayscale-ish with hue)
            unique = np.unique(data[data >= 0])
            colormap = {}
            for i, cls in enumerate(unique):
                hue = (i * 137) % 360
                from colorsys import hsv_to_rgb
                r, g, b = hsv_to_rgb(hue / 360.0, 0.6, 0.9)
                colormap[int(cls)] = (int(r * 255), int(g * 255), int(b * 255), 200)
            colormap[255] = (0, 0, 0, 0)
            if -1 in np.unique(data):
                colormap[-1] = (0, 0, 0, 0)
        else:
            colormap = CLASSIFICATION_COLORMAP
        png_bytes, _ = _raster_to_preview_png(data, colormap)
        b64 = base64.b64encode(png_bytes).decode("utf-8")

    return jsonify({"bounds": bounds_wgs84, "image_data_url": f"data:image/png;base64,{b64}"})


@app.route("/api/outputs/vitality-tile/<tile_id>", methods=["GET"])
def outputs_vitality_tile(tile_id):
    """Return bounds (WGS84) and a small PNG preview of vitality_classes.tif as data URL (lightweight overlay)."""
    aoi_suffix = request.args.get("aoi_suffix", "").strip()
    if not aoi_suffix:
        return jsonify({"error": "Missing aoi_suffix"}), 400
    try:
        sanitize_path_component(aoi_suffix)
        sanitize_path_component(tile_id)
    except ValueError:
        return jsonify({"error": "Invalid aoi_suffix or tile_id"}), 400

    base = Path(_outputs_base_dir(aoi_suffix))
    tif_path = base / "vitality_autoencoder" / f"tile_{tile_id}" / "vitality_classes.tif"

    if not tif_path.exists():
        return jsonify({"error": "Vitality tile not found"}), 404

    with rasterio.open(tif_path) as src:
        crs = src.crs
        bounds = src.bounds
        data = src.read(1)
        if crs and not getattr(crs, "is_geographic", True):
            w, s, e, n = transform_bounds(crs, "EPSG:4326", *bounds)
        else:
            w, s, e, n = bounds.left, bounds.bottom, bounds.right, bounds.top
        bounds_wgs84 = [[s, w], [n, e]]
        png_bytes, _ = _raster_to_preview_png(data, VITALITY_COLORMAP)
        b64 = base64.b64encode(png_bytes).decode("utf-8")

    return jsonify({"bounds": bounds_wgs84, "image_data_url": f"data:image/png;base64,{b64}"})


@app.route('/api/results/<step_id>', methods=['GET'])
def get_results(step_id):
    """Get results for a specific step."""
    if step_id in workflow_state["results"]:
        return jsonify(workflow_state["results"][step_id])
    return jsonify({"error": "No results found"}), 404


@app.route('/api/quality-check/<step_id>', methods=['GET'])
def get_quality_check(step_id):
    """Get quality check results for a specific step."""
    if step_id in workflow_state["steps"]:
        quality_check = workflow_state["steps"][step_id].get("quality_check")
        if quality_check:
            return jsonify(quality_check)
    return jsonify({"error": "No quality check found"}), 404


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset workflow state."""
    global workflow_state
    workflow_state = {
        "current_run": None,
        "status": "idle",
        "current_step": None,
        "steps": {},
        "logs": [],
        "results": {}
    }
    return jsonify({"message": "Workflow state reset"})


@app.route('/dashboard')
def dashboard():
    """Serve the dashboard page."""
    return send_from_directory('static', 'dashboard.html')


@app.route('/intro')
def intro():
    """Serve the intro presentation page."""
    return send_from_directory('static', 'intro.html')


@app.route('/api/project-info', methods=['GET'])
def get_project_info():
    """Get project information for intro presentation."""
    project_info = {
        "location": "Project Area",
        "country": "Unknown",
        "area": None,
        "dateRange": "N/A"
    }
    
    # Get config from workflow state
    config = workflow_state.get("last_config", {})
    if not config:
        # Try to get from any step
        for step_info in workflow_state["steps"].values():
            if "config" in step_info:
                config = step_info["config"]
                break
    
    # Also try to get from request if available (for when user hasn't run workflow yet)
    if not config or not config.get("aoi"):
        # Try to get from request params or form data
        try:
            from flask import request
            if request.args.get("aoi"):
                if not config:
                    config = {}
                config["aoi"] = request.args.get("aoi")
            if request.args.get("date"):
                if not config:
                    config = {}
                config["date"] = request.args.get("date")
            if request.args.get("window_days"):
                if not config:
                    config = {}
                config["window_days"] = int(request.args.get("window_days"))
        except:
            pass
    
    # If still no config, try to get from default UI values (this is a fallback)
    # The frontend should send this via request params
    
    if config:
        # Extract AOI name
        aoi = config.get("aoi") or config.get("input", {}).get("aoi")
        if aoi:
            # Try to extract location from AOI name
            aoi_name = aoi.split(".")[-1] if "." in aoi else aoi
            project_info["location"] = aoi_name.replace("_", " ").title()
            
            # Try to calculate area
            try:
                from treelance_sentinel.sentinel_download.sentinel_aws_downloader import _load_geometry
                from shapely.geometry import shape
                
                geom = _load_geometry(aoi)
                geom_gdf = gpd.GeoDataFrame.from_features([{
                    "type": "Feature",
                    "properties": {},
                    "geometry": geom
                }], crs="EPSG:4326")
                
                # Project to Web Mercator for area calculation
                geom_projected = geom_gdf.to_crs(epsg=3857)
                area_sqm = geom_projected.geometry.iloc[0].area
                area_km2 = area_sqm / 1_000_000
                project_info["area"] = round(area_km2, 2)
            except Exception as e:
                logger.warning(f"Failed to calculate AOI area for project info: {e}")
            
            # Try to extract country from AOI name (common patterns)
            aoi_lower = aoi_name.lower()
            if "germany" in aoi_lower or "deutschland" in aoi_lower or "bayern" in aoi_lower or "ger" in aoi_lower:
                project_info["country"] = "Germany"
            elif "austria" in aoi_lower or "osterreich" in aoi_lower or "aut" in aoi_lower:
                project_info["country"] = "Austria"
            elif "france" in aoi_lower or "fra" in aoi_lower:
                project_info["country"] = "France"
            elif "spain" in aoi_lower or "espana" in aoi_lower or "esp" in aoi_lower:
                project_info["country"] = "Spain"
            elif "italy" in aoi_lower or "italia" in aoi_lower or "ita" in aoi_lower:
                project_info["country"] = "Italy"
            elif "poland" in aoi_lower or "pol" in aoi_lower:
                project_info["country"] = "Poland"
            elif "czech" in aoi_lower or "cze" in aoi_lower:
                project_info["country"] = "Czech Republic"
            else:
                project_info["country"] = "Unknown"
        
        # Get date range - check multiple sources
        date_str = config.get("date")
        window_days = config.get("window_days", 0) or 0
        
        if date_str:
            try:
                from datetime import datetime, timedelta
                center = datetime.fromisoformat(date_str)
                if window_days > 0:
                    half = window_days // 2
                    start = center - timedelta(days=half)
                    end = center + timedelta(days=half)
                    project_info["dateRange"] = f"{start.date()} to {end.date()}"
                else:
                    project_info["dateRange"] = str(center.date())
            except Exception:
                project_info["dateRange"] = date_str
        else:
            # Fallback to sentinel_aws.time_range
            sentinel_cfg = config.get("sentinel_aws", {})
            time_range = sentinel_cfg.get("time_range", "")
            if time_range:
                project_info["dateRange"] = time_range.replace("/", " to ")
            else:
                project_info["dateRange"] = "Not specified"
    
    return jsonify(project_info)


@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    """Get aggregated dashboard data from all quality checks."""
    dashboard_data = {
        "steps": {},
        "total_processing_time": 0,
        "aoi_area_km2": None,
        "processing_time_per_100km2": None,
        "workflow_status": workflow_state["status"],
        "project_info": {}
    }
    
    # Get project info from config (if available from last run)
    # Try to get from workflow state or config
    config = getattr(workflow_state, 'last_config', {})
    if not config:
        # Try to get from any step's config
        for step_info in workflow_state["steps"].values():
            if "config" in step_info:
                config = step_info["config"]
                break
    
    # Calculate AOI area
    aoi_area_km2 = None
    if config:
        aoi = config.get("aoi") or config.get("input", {}).get("aoi")
        if aoi:
            try:
                # Try to load AOI and calculate area
                from treelance_sentinel.sentinel_download.sentinel_aws_downloader import _load_geometry
                from shapely.geometry import shape
                
                geom = _load_geometry(aoi)
                geom_shape = shape(geom)
                
                # Convert to Web Mercator for accurate area calculation
                geom_gdf = gpd.GeoDataFrame.from_features([{
                    "type": "Feature",
                    "properties": {},
                    "geometry": geom
                }], crs="EPSG:4326")
                
                # Project to Web Mercator (EPSG:3857) for area in square meters
                geom_projected = geom_gdf.to_crs(epsg=3857)
                area_sqm = geom_projected.geometry.iloc[0].area
                aoi_area_km2 = area_sqm / 1_000_000  # Convert to km²
                
                dashboard_data["aoi_area_km2"] = round(aoi_area_km2, 2)
                
                # Extract project info
                aoi_name = aoi.split(".")[-1] if "." in aoi else aoi
                dashboard_data["project_info"]["aoi_name"] = aoi_name
                
                # Get date range from config
                sentinel_cfg = config.get("sentinel_aws", {})
                time_range = sentinel_cfg.get("time_range", "")
                if time_range:
                    dashboard_data["project_info"]["time_range"] = time_range
                
            except Exception as e:
                logger.warning(f"Failed to calculate AOI area: {e}")
    
    # Aggregate data from all steps
    total_time = 0
    for step_id, step_info in workflow_state["steps"].items():
        if "quality_check" in step_info:
            dashboard_data["steps"][step_id] = {
                "quality_check": step_info["quality_check"]
            }
            if step_info["quality_check"].get("duration_minutes"):
                total_time += step_info["quality_check"]["duration_minutes"]
    
    dashboard_data["total_processing_time"] = round(total_time, 1)
    
    # Calculate normalized processing time per 100km²
    if aoi_area_km2 and aoi_area_km2 > 0:
        processing_time_per_100km2 = (total_time / aoi_area_km2) * 100
        dashboard_data["processing_time_per_100km2"] = round(processing_time_per_100km2, 1)
    else:
        # If we can't calculate area, use total time (fallback)
        dashboard_data["processing_time_per_100km2"] = round(total_time, 1)
    
    return jsonify(dashboard_data)


def get_default_config() -> dict:
    """Return default configuration dictionary (independent of any YAML file)."""
    return {
        "directories": {
            "base_output_dir": "",  # Will be auto-set from AOI name
            "raw_data": "raw_data",
            "processed_data": "processed_data",
            "asset_preparation": "asset_preparation",
            "segmentation": "segmentation",
            "predictions": "predictions",
            "logs": "logs",
        },
        "input": {
            "buffer_distance": 200,
            "segment_length": 50,
        },
        "sentinel_aws": {
            "time_range": "2025-05-01/2025-05-30",  # Placeholder, overwritten by date/window_days
            "bands": ["B02", "B03", "B04", "B05", "B06", "B08", "B11", "B12"],
            "max_cloud_cover": 10,
            "pick_lowest_cloud": True,  # Default: checked
            "clip_to_aoi": True,  # Default: checked
        },
        "processing": {
            "redo": False,
            "clear_cache": False,
            "tile_size": 512,
            "overlap": 0.1,
            "num_workers": 14,
            "kmeans": {
                # Do NOT set num_classes here: use NDVI-split (5 low + 30 high) for initial segmentation
                "high_ndvi_clusters": 30,
                "low_ndvi_clusters": 5,
                "ndvi_threshold": 0.2,
                "sample_fraction": 0.1,
                "tree_clustering": {
                    "num_classes": 8,
                    "use_pca": True,
                    "pca_components": 3,
                    "use_hdbscan": False,
                    "hdbscan_min_cluster_size": 600,
                    "hdbscan_min_samples": 15,
                    "hdbscan_cluster_selection_epsilon": 0.05,
                    "hdbscan_cluster_selection_method": "leaf",
                },
            },
            "zonal_stats_optimization": {
                # Use all 8 bands (B02–B08, B11, B12) for training features:
                # 1→B02, 2→B03, 3→B04, 4→B05, 5→B06, 6→B08, 7→B11, 8→B12
                "band_whitelist": [1, 2, 3, 4, 5, 6, 7, 8],
                "ops": ["mean"],
            },
        },
        "prediction": {
            "mode": "load",
            "model_load_path": "model_checkpoints/sentinel_model_finetuned__ts-20260104__f1-0.9797__ft-11.pt",
            "model_save_path": "model_checkpoints/sentinel_model_finetuned.pt",
            "batch_size": 1000,
            "gpu_max_rows": 1000000,
            "memory_limit_mb": 32000,
            "retraining": {
                "enabled": True,
                "epochs": 100,
                "patience": 20,
                "samples_per_class": 1000,  # Keep lowest N confidence predictions per class (tree, urban, grass) for balanced labeling
                "kill_other_gpu_python_processes": True,
                "kill_other_gpu_python_processes_grace_seconds": 8,
                "enable_gpu_monitoring": True,  # Enable background nvidia-smi monitoring
                "gpu_monitor_interval": 5,  # Update interval in seconds
            },
        },
        "vitality": {
            "epochs": 14,  # Default: 14 for Conv-Transformer (stronger regularization)
            "lr": 1e-3,
            "batch_size": 262144,
            "max_train_pixels": 300000,
            "severe_morbidity_percentile": 0.95,
            "medium_morbidity_percentile": 0.85,
            "no_change_abs_delta_ndvi": 0.03,  # Default: 0.03 (3% change threshold)
            "no_change_abs_delta_ndmi": 0.03,  # Default: 0.03 (3% change threshold)
            "no_change_percentile": 0.70,  # Default: 0.70 (70% no-change, 5% severe, 10% medium, 15% vitality)
            "attention_heads": "both",
            # Base band names – each is used for BOTH time steps.
            # Default: Recommended raw bands for dual-branch architecture:
            # - Spatial Branch (Conv): B02, B03, B04, B08 (texture, canopy structure)
            # - Spectral Branch (Transformer): B05, B11, B12 (physiology, moisture)
            # Note: While architecture doesn't fully separate branches, spectral attention prioritizes B05/B11
            "feature_bands": ["B02", "B03", "B04", "B05", "B08", "B11", "B12"],
        },
        "logging": {
            "log_level": "INFO",
        },
    }


@app.route('/api/config-template', methods=['GET'])
def config_template():
    """Return selected defaults for the Advanced UI (independent of YAML files)."""
    defaults = get_default_config()
    sentinel_aws = defaults.get("sentinel_aws", {})
    processing = defaults.get("processing", {})
    kmeans = processing.get("kmeans", {})
    tree_clustering = kmeans.get("tree_clustering", {})
    prediction = defaults.get("prediction", {})
    retraining = prediction.get("retraining", {})
    
    return jsonify(
        {
            "sentinel_aws": {
                "bands": sentinel_aws.get("bands", []),
                "max_cloud_cover": sentinel_aws.get("max_cloud_cover"),
                "pick_lowest_cloud": sentinel_aws.get("pick_lowest_cloud"),
                "clip_to_aoi": sentinel_aws.get("clip_to_aoi"),
            },
            "processing": {
                "tile_size": processing.get("tile_size"),
                "overlap": processing.get("overlap"),
                "num_workers": processing.get("num_workers"),
                "redo": processing.get("redo"),
                "clear_cache": processing.get("clear_cache"),
                "kmeans": {
                    "num_classes": kmeans.get("num_classes"),
                    "high_ndvi_clusters": kmeans.get("high_ndvi_clusters"),
                    "low_ndvi_clusters": kmeans.get("low_ndvi_clusters"),
                    "tree_clustering": {
                        "use_pca": tree_clustering.get("use_pca"),
                        "pca_components": tree_clustering.get("pca_components"),
                        "use_hdbscan": tree_clustering.get("use_hdbscan"),
                        "hdbscan_min_cluster_size": tree_clustering.get("hdbscan_min_cluster_size"),
                        "hdbscan_min_samples": tree_clustering.get("hdbscan_min_samples"),
                        "hdbscan_cluster_selection_epsilon": tree_clustering.get("hdbscan_cluster_selection_epsilon"),
                        "hdbscan_cluster_selection_method": tree_clustering.get("hdbscan_cluster_selection_method"),
                    },
                },
            },
            "prediction": {
                "model_load_path": prediction.get("model_load_path"),
                "model_save_path": prediction.get("model_save_path"),
                "batch_size": prediction.get("batch_size"),
                "retraining": {
                    "enabled": retraining.get("enabled"),
                    "epochs": retraining.get("epochs"),
                    "patience": retraining.get("patience"),
                },
            },
            "vitality": {
                "epochs": defaults.get("vitality", {}).get("epochs", 14),  # Default: 14 for Conv-Transformer
                "lr": defaults.get("vitality", {}).get("lr", 1e-3),
                "batch_size": defaults.get("vitality", {}).get("batch_size", 262144),
                "max_train_pixels": defaults.get("vitality", {}).get("max_train_pixels", 300000),
                "severe_morbidity_percentile": defaults.get("vitality", {}).get("severe_morbidity_percentile", 0.95),
                "medium_morbidity_percentile": defaults.get("vitality", {}).get("medium_morbidity_percentile", 0.85),
                "attention_heads": defaults.get("vitality", {}).get("attention_heads", "both"),
                # Base band names – UI checkboxes use these, and the backend
                # expands them to both time steps.
                # Default: Recommended raw bands for dual-branch architecture:
                # - Spatial Branch (Conv): B02, B03, B04, B08 (texture, canopy structure)
                # - Spectral Branch (Transformer): B05, B11, B12 (physiology, moisture)
                "feature_bands": defaults.get("vitality", {}).get("feature_bands", ["B02", "B03", "B04", "B05", "B08", "B11", "B12"]),
            },
        }
    )


def execute_workflow_step(step_id, step_config, config, workflow):
    """Execute a single workflow step."""
    workflow_state["status"] = "running"
    workflow_state["current_step"] = step_id
    workflow_state["current_run"] = workflow
    workflow_state["last_config"] = config  # Store config for dashboard
    workflow_state["steps"][step_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "logs": [],
        "config": config  # Store config in step
    }
    
    try:
        # Build command based on step and workflow
        cmd = build_command(step_id, config, workflow)
        
        # Get project root for working directory
        project_root = Path(__file__).parent.parent
        
        # Prepare environment with PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        
        # Execute command from project root
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(project_root),
            env=env
        )
        
        # Stream logs (handle both empty lines and actual content)
        for line in process.stdout:
            if line:  # Only process non-empty lines
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "step": step_id,
                    "message": line.strip()
                }
                workflow_state["logs"].append(log_entry)
            workflow_state["steps"][step_id]["logs"].append(log_entry)
        
        process.wait()
        
        # Calculate duration
        if "started_at" in workflow_state["steps"][step_id]:
            start_time = datetime.fromisoformat(workflow_state["steps"][step_id]["started_at"])
            end_time = datetime.now()
            duration_minutes = (end_time - start_time).total_seconds() / 60.0
            workflow_state["steps"][step_id]["duration"] = round(duration_minutes, 1)
        
        if process.returncode == 0:
            workflow_state["steps"][step_id]["status"] = "completed"
            workflow_state["steps"][step_id]["completed_at"] = datetime.now().isoformat()
            # Generate quality check
            try:
                quality_check = generate_quality_check(step_id, workflow_state["steps"][step_id], config)
                workflow_state["steps"][step_id]["quality_check"] = quality_check
            except Exception as e:
                logger.warning(f"Failed to generate quality check for {step_id}: {e}")
            # Collect results
            workflow_state["results"][step_id] = collect_results(step_id, config)
            # Add quality check to results
            if "quality_check" in workflow_state["steps"][step_id]:
                workflow_state["results"][step_id]["quality_check"] = workflow_state["steps"][step_id]["quality_check"]
        else:
            workflow_state["steps"][step_id]["status"] = "error"
            workflow_state["steps"][step_id]["error"] = f"Process exited with code {process.returncode}"
        
    except Exception as e:
        workflow_state["steps"][step_id]["status"] = "error"
        workflow_state["steps"][step_id]["error"] = str(e)
        # Calculate duration even on error
        if "started_at" in workflow_state["steps"][step_id]:
            start_time = datetime.fromisoformat(workflow_state["steps"][step_id]["started_at"])
            end_time = datetime.now()
            duration_minutes = (end_time - start_time).total_seconds() / 60.0
            workflow_state["steps"][step_id]["duration"] = round(duration_minutes, 1)
    
    finally:
        # Clear current_step when this step finishes
        if workflow_state.get("current_step") == step_id:
            workflow_state["current_step"] = None
        # Check if all steps are done
        if all(s.get("status") in ["completed", "error"] for s in workflow_state["steps"].values()):
            workflow_state["status"] = "completed" if all(s.get("status") == "completed" for s in workflow_state["steps"].values()) else "error"


def execute_full_workflow(config, workflow):
    """Execute all workflow steps sequentially."""
    workflow_state["status"] = "running"
    
    steps_for_workflow = WORKFLOW_STEPS.get(workflow, [])
    for step in steps_for_workflow:
        step_id = step["id"]
        execute_workflow_step(step_id, step, config, workflow)
        
        # If step failed, stop workflow
        if workflow_state["steps"][step_id]["status"] == "error":
            workflow_state["status"] = "error"
            break
    
    if workflow_state["status"] == "running":
        workflow_state["status"] = "completed"


def _prepare_config_path(config: dict, workflow: str) -> str:
    """Prepare a temporary config file based on defaults + main parameters + advanced overrides.
    
    - Uses hardcoded defaults (independent of any YAML file)
    - Overlays AOI and date from the top-level config (aoi/date/window_days)
    - Applies advanced overrides from the UI
    - Derives base_output_dir from AOI name
    """
    # Get the project root directory (parent of web_frontend)
    project_root = Path(__file__).parent.parent
    
    # Start with default config (independent of YAML files)
    base_cfg = get_default_config()
    
    # Overlay AOI and derive base_output_dir from AOI name (or from aoi_suffix for drawn rectangle)
    aoi = config.get("aoi")
    aoi_suffix_override = config.get("aoi_suffix")
    if aoi:
        base_cfg.setdefault("input", {})
        base_cfg["input"]["aoi"] = aoi
        
        # Use explicit aoi_suffix when provided (e.g. custom_rectangle_<ts> for drawn AOI)
        if aoi_suffix_override:
            try:
                aoi_suffix = sanitize_path_component(str(aoi_suffix_override))
            except ValueError as e:
                logger.warning(f"Invalid aoi_suffix '{aoi_suffix_override}': {e}. Using 'unknown'.")
                aoi_suffix = "unknown"
        else:
            # Derive S3 folder name: file path -> stem; schema.table -> last segment; else as-is
            aoi_str = str(aoi)
            if "/" in aoi_str or aoi_str.startswith("/") or os.path.exists(aoi_str):
                aoi_suffix = Path(aoi).stem
            elif "." in aoi_str:
                aoi_suffix = aoi_str.split(".")[-1]  # e.g. prediction_ready_aoi.bayernwerk_eon_one_100 -> bayernwerk_eon_one_100
            else:
                aoi_suffix = aoi_str
            try:
                aoi_suffix = sanitize_path_component(aoi_suffix)
            except ValueError as e:
                logger.warning(f"Invalid AOI name '{aoi}': {e}. Using 'unknown' as fallback.")
                aoi_suffix = "unknown"
        
        base_cfg.setdefault("directories", {})
        base_cfg["directories"]["base_output_dir"] = str(Path(LOCAL_OUTPUTS_ROOT) / aoi_suffix)
    
    # Apply structured advanced overrides (from Advanced UI), if provided
    overrides = config.get("advanced_overrides") or {}
    if overrides:
        # Sentinel AWS
        sentinel_over = overrides.get("sentinel_aws") or {}
        if sentinel_over:
            base_cfg.setdefault("sentinel_aws", {})
            base_cfg["sentinel_aws"].update(
                {k: v for k, v in sentinel_over.items() if v is not None}
            )
        
        # Processing / kmeans / tree_clustering
        proc_over = overrides.get("processing") or {}
        if proc_over:
            base_cfg.setdefault("processing", {})
            # Shallow keys
            for key in ["tile_size", "overlap", "num_workers", "redo", "clear_cache"]:
                if key in proc_over and proc_over[key] is not None:
                    base_cfg["processing"][key] = proc_over[key]
            
            # KMeans sub-config
            kmeans_over = proc_over.get("kmeans") or {}
            if kmeans_over:
                base_cfg["processing"].setdefault("kmeans", {})
                base_cfg["processing"]["kmeans"].update(
                    {
                        k: v
                        for k, v in kmeans_over.items()
                        if k != "tree_clustering" and v is not None
                    }
                )
                tree_over = kmeans_over.get("tree_clustering") or {}
                if tree_over:
                    base_cfg["processing"]["kmeans"].setdefault("tree_clustering", {})
                    base_cfg["processing"]["kmeans"]["tree_clustering"].update(
                        {k: v for k, v in tree_over.items() if v is not None}
                    )
        
        # Prediction / retraining
        pred_over = overrides.get("prediction") or {}
        if pred_over:
            base_cfg.setdefault("prediction", {})
            # Model paths
            if "model_load_path" in pred_over and pred_over["model_load_path"]:
                base_cfg["prediction"]["model_load_path"] = pred_over["model_load_path"]
            if "model_save_path" in pred_over and pred_over["model_save_path"]:
                base_cfg["prediction"]["model_save_path"] = pred_over["model_save_path"]
            # Batch size
            if "batch_size" in pred_over and pred_over["batch_size"] is not None:
                base_cfg["prediction"]["batch_size"] = pred_over["batch_size"]
            retr_over = pred_over.get("retraining") or {}
            if retr_over:
                base_cfg["prediction"].setdefault("retraining", {})
                base_cfg["prediction"]["retraining"].update(
                    {k: v for k, v in retr_over.items() if v is not None}
                )
    
    # For single-date (normal) mode, set explicit time_range around the date
    date_str = config.get("date")
    window_days = int(config.get("window_days", 0) or 0)
    if date_str:
        base_cfg.setdefault("sentinel_aws", {})
        if workflow == "normal":
            # Use date +/- window_days/2 if provided, else exact date
            try:
                from datetime import datetime, timedelta
                center = datetime.fromisoformat(date_str)
                if window_days > 0:
                    half = window_days // 2
                    start = center - timedelta(days=half)
                    end = center + timedelta(days=half)
                else:
                    start = end = center
                base_cfg["sentinel_aws"]["time_range"] = f"{start.date()}/{end.date()}"
            except Exception:
                # Fallback: simple date/date range
                base_cfg["sentinel_aws"]["time_range"] = f"{date_str}/{date_str}"
    
    # Force NDVI-split for initial segmentation (5 low + 30 high): remove num_classes so
    # imagery_processing.perform_kmeans uses high_ndvi_clusters + low_ndvi_clusters, not single K-means
    base_cfg.setdefault("processing", {})
    base_cfg["processing"].setdefault("kmeans", {})
    kmeans_cfg = base_cfg["processing"]["kmeans"]
    if isinstance(kmeans_cfg, dict):
        kmeans_cfg.pop("num_classes", None)
        kmeans_cfg.setdefault("high_ndvi_clusters", 30)
        kmeans_cfg.setdefault("low_ndvi_clusters", 5)

    # Write to a temporary YAML under project_root/tmp_configs
    tmp_dir = project_root / "tmp_configs"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"treelance_config_{workflow}_{int(time.time())}.yaml"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(base_cfg, f, sort_keys=False)
    
    return str(tmp_path)


def _filter_logs_for_step(step_id, all_logs):
    """Filter logs to only include messages relevant to the specific step.
    
    This is necessary because some CLI operations run multiple steps,
    and we want to associate logs with the correct frontend step.
    """
    if not all_logs:
        return []
    
    # Define log markers that indicate the start/end of each step
    # Using more specific patterns to avoid false matches
    step_markers = {
        "download": {
            "start": [r"download.*sentinel", r"searching.*sentinel", r"found.*items.*time range"],
            "end": [r"downloaded.*successfully", r"saved.*stack", r"download.*complete"],
            "include": [r"download", r"sentinel", r"scene", r"cloud_cover", r"aoi_coverage", r"selected.*scene"]
        },
        "preprocess": {
            "start": [r"asset preparation", r"preparing.*asset", r"process.*imagery"],
            "end": [r"processed.*imagery", r"imagery.*complete", r"stacked.*bands"],
            "include": [r"asset", r"tile.*process", r"process.*imagery", r"stack", r"reproject", r"resample"]
        },
        "segmentation": {
            "start": [r"segmentation", r"k-means.*segmentation", r"running.*k-means"],
            "end": [r"segmentation.*complete", r"k-means.*complete", r"saved.*polygons", r"polygonization.*complete"],
            "include": [r"segmentation", r"k-means", r"cluster.*segmentation", r"polygon", r"polygonization"]
        },
        "prediction": {
            "start": [r"making predictions", r"model.*prediction", r"predicting"],
            "end": [r"prediction.*complete", r"saved.*predictions", r"classification.*summary", r"total predictions"],
            "include": [r"prediction", r"model", r"confidence", r"classification", r"total predictions", r"processed.*file"]
        },
        "change_detection": {
            "start": [r"organizing.*time series.*change detection", r"change detection", r"📊.*change detection"],
            "end": [r"change detection.*complete", r"change detection organization complete"],
            "include": [r"change detection", r"organizing.*time series", r"time_step_[12]", r"applying.*tree mask", r"tile.*time step"]
        },
        "tree_clustering": {
            "start": [r"tree.*clustering", r"performing.*tree.*clustering", r"🎯.*tree clustering", r"tree-specific.*clustering"],
            "end": [r"tree.*clustering.*complete", r"tree-specific.*complete", r"clustering complete.*cluster distribution"],
            "include": [r"tree.*cluster", r"k-means.*cluster", r"hdbscan", r"found.*clusters", r"cluster.*distribution", r"valid pixels.*clustering"]
        },
        "vitality": {
            "start": [r"vitality", r"autoencoder", r"training.*autoencoder", r"💚.*vitality"],
            "end": [r"vitality.*complete", r"autoencoder.*complete", r"uploaded.*vitality", r"summary\.json"],
            "include": [r"vitality", r"autoencoder", r"morbidity", r"vitality.*class", r"class_rates", r"severe_morbidity", r"medium_morbidity"]
        }
    }
    
    markers = step_markers.get(step_id, {})
    if not markers:
        # No filtering - return all logs
        return all_logs
    
    filtered_logs = []
    in_step = False
    start_patterns = [re.compile(p, re.IGNORECASE) for p in markers.get("start", [])]
    end_patterns = [re.compile(p, re.IGNORECASE) for p in markers.get("end", [])]
    include_patterns = [re.compile(p, re.IGNORECASE) for p in markers.get("include", [])]
    
    # Track if we've seen the end marker (to include a few more logs after)
    end_seen = False
    logs_after_end = 0
    
    for log_entry in all_logs:
        msg = log_entry.get("message", "")
        
        # Check if we're entering this step
        if not in_step:
            for pattern in start_patterns:
                if pattern.search(msg):
                    in_step = True
                    filtered_logs.append(log_entry)
                    end_seen = False
                    logs_after_end = 0
                    break
        else:
            # We're in the step - include this log
            filtered_logs.append(log_entry)
            
            # Check if we're leaving this step
            if not end_seen:
                for pattern in end_patterns:
                    if pattern.search(msg):
                        end_seen = True
                        logs_after_end = 0
                        break
            
            # Include a few more logs after the end marker (up to 5)
            if end_seen:
                logs_after_end += 1
                if logs_after_end > 5:
                    break
    
    # If no logs matched start/end patterns, try to include logs that match include patterns
    if not filtered_logs:
        for log_entry in all_logs:
            msg = log_entry.get("message", "")
            for pattern in include_patterns:
                if pattern.search(msg):
                    filtered_logs.append(log_entry)
                    break
    
    # If still no logs, return all logs (better than nothing)
    return filtered_logs if filtered_logs else all_logs


def build_command(step_id: str, config: dict, workflow: str) -> list[str]:
    """Build CLI command for a workflow step and workflow type.
    
    - Normal workflow uses classic vegbins mode + --steps
    - Time-series workflow uses time_series mode + --time-series-step/--time-series-steps
    
    Security: All inputs are validated before being used in the command.
    
    Args:
        step_id: The step identifier (validated against whitelist)
        config: Configuration dictionary
        workflow: Workflow type ('normal' or 'time_series')
        
    Returns:
        List of command arguments (safe for subprocess.Popen with shell=False)
        
    Raises:
        ValueError: If any input fails validation
    """
    # Validate workflow type
    if not validate_workflow(workflow):
        raise ValueError(f"Invalid workflow type: {workflow}")
    
    # Validate step_id against whitelist
    if step_id is not None and not validate_step_id(step_id, workflow):
        raise ValueError(f"Invalid step_id '{step_id}' for workflow '{workflow}'")
    
    # Validate date format if provided
    date_str = config.get("date", "")
    if date_str and not validate_date_format(date_str):
        raise ValueError(f"Invalid date format: {date_str}")
    
    # Use the Python from the conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        python_cmd = os.path.join(conda_prefix, "bin", "python")
    else:
        python_cmd = sys.executable  # Use current Python interpreter
    
    # Prepare a config file path (includes AOI/date overrides and advanced YAML)
    config_path = _prepare_config_path(config, workflow)
    
    # Normal single-date workflow (classic vegbins)
    if workflow == "normal":
        base_cmd = [
            python_cmd, "-m", "treelance_sentinel.cli",
            "--config", config_path,
            "--mode", "vegbins",
        ]
        
        # Map step IDs to vegbins --steps
        # STEPS in cli.py: 1=Download Sentinel Data, 2=Asset Preparation,
        # 3=Process Imagery, 4=Segmentation, 5=Model Prediction, 6=Create Folium Map
        normal_step_mapping = {
            "normal_download": "1",
            "normal_asset_prep": "2",
            "normal_process_imagery": "3",
            "normal_segmentation": "4",
            "normal_prediction": "5",
            "normal_folium": "6",
        }
        step_number = normal_step_mapping.get(step_id)
        if step_number:
            base_cmd.extend(["--steps", step_number])
    
    else:
        # Time-series workflow
        # Map each frontend step to the correct CLI step numbers and time-series-step mode
        # CLI steps: 1=Download, 2=Asset Prep, 3=Process Imagery, 4=Segmentation, 5=Prediction
        # Time-series modes: confidence, finetune_predict, vitality_only, all_vitality
        
        if step_id == "all" or step_id is None:
            time_series_step = "all_vitality"
            cli_steps = None  # Run all steps
            run_vitality = False  # all_vitality mode handles vitality internally
        else:
            # Map frontend step IDs to CLI step numbers and time-series-step mode
            step_configs = {
            "download": {
                    "time_series_step": "confidence",
                    "cli_steps": "1",  # Only download
            },
            "preprocess": {
                "time_series_step": "confidence",
                    "cli_steps": "2,3",  # Asset prep + process imagery (skip download if already done)
            },
            "segmentation": {
                "time_series_step": "confidence",
                    "cli_steps": "4",  # Only segmentation (assumes download/preprocess already done)
            },
            "prediction": {
                    "time_series_step": "finetune_predict",
                    "cli_steps": "5",  # Only prediction (assumes previous steps done)
            },
            "change_detection": {
                    "time_series_step": "confidence",
                    "cli_steps": None,  # change_detection is a separate function, not a numbered step
                    "run_vitality": True,  # This triggers change_detection + tree_clustering in confidence mode
            },
            "tree_clustering": {
                    "time_series_step": "confidence",
                    "cli_steps": None,  # tree_clustering is a separate function, not a numbered step
                    "run_vitality": True,  # This triggers change_detection + tree_clustering in confidence mode
            },
            "vitality": {
                    "time_series_step": "vitality_only",
                    "cli_steps": None,  # vitality_only skips all numbered steps
                },
            }
            
            step_config = step_configs.get(step_id, {"time_series_step": "all_vitality", "cli_steps": None, "run_vitality": False})
            time_series_step = step_config["time_series_step"]
            cli_steps = step_config["cli_steps"]
            run_vitality = step_config.get("run_vitality", False)
        
        base_cmd = [
            python_cmd, "-m", "treelance_sentinel.cli",
            "--config", config_path,
            "--mode", "time_series",
            "--time-series-date", config.get("date", "2025-06-01"),
            "--time-series-window-days", str(config.get("window_days", 45)),
            "--time-series-step", time_series_step,
            "--time-series-tree-mask-source", config.get("tree_mask_source", "previous"),
            "--time-series-label-source", config.get("label_source", "previous"),
            "--time-series-snapshots", config.get("snapshots", "both"),
        ]
        
        # Add temporal mode (applies to all time_series steps, not just download)
        temporal_mode = config.get("temporal_mode", "bi-temporal")
        if temporal_mode == "tri-temporal":
            base_cmd.extend(["--time-series-temporal-mode", "tri-temporal"])
            if config.get("pre_greenup_date"):
                base_cmd.extend(["--time-series-pre-greenup-date", config["pre_greenup_date"]])
        
        # Add step-specific CLI step numbers if specified
        if cli_steps:
            base_cmd.extend(["--time-series-steps", cli_steps])
        
        # For change_detection and tree_clustering, we need to trigger them via run_vitality flag
        # This makes confidence mode run change_detection + tree_clustering
        if run_vitality:
            # The CLI checks for time_series_run_vitality attribute, but we can't set that via command line
            # Instead, we use confidence_vitality mode which triggers change_detection + tree_clustering
            # Find the index of --time-series-step argument and update its value
            try:
                step_idx = base_cmd.index("--time-series-step")
                base_cmd[step_idx + 1] = "confidence_vitality"
            except ValueError:
                # Fallback: if --time-series-step not found, append it
                base_cmd.extend(["--time-series-step", "confidence_vitality"])
    
    # Add vitality-specific arguments if provided (only for time_series workflow)
    if workflow == "time_series":
        if config.get("vitality_epochs"):
            base_cmd.extend(["--vitality-epochs", str(config["vitality_epochs"])])
        if config.get("vitality_lr"):
            base_cmd.extend(["--vitality-lr", str(config["vitality_lr"])])
        if config.get("vitality_batch_size"):
            base_cmd.extend(["--vitality-batch-size", str(config["vitality_batch_size"])])
        if config.get("vitality_max_train_pixels"):
            base_cmd.extend(["--vitality-max-train-pixels", str(config["vitality_max_train_pixels"])])
        # Always include no_change thresholds (default: 0.03 if not specified)
        no_change_ndvi = config.get("vitality_no_change_abs_delta_ndvi", 0.03)
        no_change_ndmi = config.get("vitality_no_change_abs_delta_ndmi", 0.03)
        no_change_percentile = config.get("vitality_no_change_percentile", 0.70)
        base_cmd.extend(["--vitality-no-change-abs-delta-ndvi", str(no_change_ndvi)])
        base_cmd.extend(["--vitality-no-change-abs-delta-ndmi", str(no_change_ndmi)])
        base_cmd.extend(["--vitality-no-change-percentile", str(no_change_percentile)])
        if config.get("vitality_severe_morbidity_percentile") is not None:
            base_cmd.extend(["--vitality-severe-morbidity-percentile", str(config["vitality_severe_morbidity_percentile"])])
        if config.get("vitality_medium_morbidity_percentile") is not None:
            base_cmd.extend(["--vitality-medium-morbidity-percentile", str(config["vitality_medium_morbidity_percentile"])])
        if config.get("vitality_feature_bands"):
            base_cmd.extend(["--vitality-feature-bands"] + config["vitality_feature_bands"])
        if config.get("vitality_attention_heads"):
            base_cmd.extend(["--vitality-attention-heads", config["vitality_attention_heads"]])
        
        # Treelance 2.0 features - Temporal mode (inherited from main config)
        # Note: Spectral Attention, Weighted Loss, and Deep Feature Differencing are enabled by default
        # in the autoencoder code. The CLI doesn't have flags to disable them, but we can pass
        # feature bands and other parameters that affect their behavior.
        temporal_mode = config.get("temporal_mode", "bi-temporal")
        if temporal_mode == "tri-temporal":
            # Tri-temporal mode: use provided time_step_0_s3 or auto-generate from pre_greenup_date
            if config.get("vitality_time_step_0_s3"):
                base_cmd.extend(["--vitality-time-step-0-s3", config["vitality_time_step_0_s3"]])
            # If not provided, time_step_0 will be auto-generated from pre_greenup_date during change_detection step
        # Note: Spectral Attention, Weighted Loss, and Deep Feature Differencing are always enabled
        # in the current implementation. The checkboxes in the UI are informational only.
    
    if config.get("overwrite_cache"):
        base_cmd.append("--overwrite-cache")
    
    return base_cmd


def generate_quality_check(step_id, step_info, config):
    """Generate quality check JSON for a completed step and save to S3."""
    import re
    import json as json_lib
    
    # Find step name from WORKFLOW_STEPS
    step_name = step_id
    for workflow_steps in WORKFLOW_STEPS.values():
        for step in workflow_steps:
            if step["id"] == step_id:
                step_name = step.get("name", step_id)
                break
    
    quality_check = {
        "step_id": step_id,
        "step_name": step_name,
        "status": step_info.get("status", "unknown"),
        "started_at": step_info.get("started_at"),
        "completed_at": step_info.get("completed_at"),
        "duration_minutes": step_info.get("duration"),
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
        "files": {},
        "logs_summary": []
    }
    
    # Parse logs to extract metrics
    all_logs = step_info.get("logs", [])
    
    # Filter logs to only include those relevant to this step
    # This is important because some steps trigger multiple CLI operations
    logs = _filter_logs_for_step(step_id, all_logs)
    
    # Extract information based on step type
    if step_id == "download" or step_id == "normal_download":
        # Only show scenes that were actually chosen for download (not all candidates).
        # Downloader logs "Tile X: Selected <scene_id> (cloud: ...%)" for each chosen scene;
        # it also logs "- <scene_id> | ..." for every candidate (do not use those unless we
        # confirm the scene was chosen via a "Selected <id> (cloud:" line).
        chosen_ids = set()
        selected_scenes = []
        for log in logs:
            msg = log.get("message", "")
            selected_match = re.search(r"selected\s+(\S+)\s+\(cloud:\s*([\d.]+)%", msg, re.IGNORECASE)
            if selected_match:
                scene_id = selected_match.group(1)
                chosen_ids.add(scene_id)
                cloud_pct = float(selected_match.group(2))
                aoi_cov = None
                for log2 in logs:
                    if scene_id in log2.get("message", ""):
                        cov_match = re.search(r"aoi_coverage=([\d.]+)%", log2.get("message", ""), re.IGNORECASE)
                        if cov_match:
                            aoi_cov = float(cov_match.group(1))
                            break
                selected_scenes.append({
                    "scene_id": scene_id,
                    "cloud_cover_percent": cloud_pct,
                    "aoi_coverage_percent": aoi_cov,
                })
        # Fallback: if no "Selected X (cloud:" lines matched, parse listing but only add scenes that appear in a "Selected X (cloud:" line
        if not selected_scenes:
            for log in logs:
                msg = log.get("message", "")
                selected_match = re.search(r"selected\s+(\S+)\s+\(cloud:", msg, re.IGNORECASE)
                if selected_match:
                    chosen_ids.add(selected_match.group(1))
            for log in logs:
                msg = log.get("message", "")
                scene_match = re.search(
                    r"-\s+(\S+)\s+\|\s+.*?cloud_cover=([\d.]+)%(?:\s+\|\s+aoi_coverage=([\d.]+)%)?",
                    msg,
                    re.IGNORECASE,
                )
                if scene_match:
                    scene_id = scene_match.group(1)
                    if scene_id not in chosen_ids:
                        continue
                    cloud_pct = float(scene_match.group(2))
                    aoi_cov = float(scene_match.group(3)) if scene_match.group(3) else None
                    selected_scenes.append({
                        "scene_id": scene_id,
                        "cloud_cover_percent": cloud_pct,
                        "aoi_coverage_percent": aoi_cov,
                    })
        quality_check["metrics"] = {
            "selected_scenes": selected_scenes
        }
    
    elif step_id == "preprocess" or step_id == "normal_process_imagery":
        # Look for processing metrics
        tiles_processed = 0
        pixels_processed = 0
        
        for log in logs:
            msg = log.get("message", "")
            msg_lower = msg.lower()
            # Look for "Processing complete. Successfully processed: X, Failed: Y"
            # or "Processed {count} tiles" or "Successfully processed: {count}"
            success_match = re.search(r'successfully\s+processed[:\s]+(\d+)', msg_lower, re.IGNORECASE)
            if success_match:
                tiles_processed = max(tiles_processed, int(success_match.group(1)))
            # Look for tile count: "X tiles" or "X tile"
            tile_match = re.search(r'(\d+)\s+tile[s]?', msg_lower, re.IGNORECASE)
            if tile_match:
                tiles_processed = max(tiles_processed, int(tile_match.group(1)))
            # Look for pixel count: "X pixels" or "X pixel"
            pixel_match = re.search(r'(\d+[,.]?\d*)\s+pixel[s]?', msg_lower, re.IGNORECASE)
            if pixel_match:
                pixel_str = pixel_match.group(1).replace(',', '').replace('.', '')
                if pixel_str.isdigit():
                    pixels_processed = max(pixels_processed, int(pixel_str))
        
        quality_check["metrics"] = {
            "tiles_processed": tiles_processed,
            "pixels_processed": pixels_processed
        }
    
    elif step_id == "segmentation" or step_id == "normal_segmentation":
        # Look for segmentation metrics
        polygons_created = 0
        clusters_used = 0
        
        for log in logs:
            msg = log.get("message", "")
            msg_lower = msg.lower()
            # Look for "✅ Saved {count:,} polygons to {path}"
            poly_match = re.search(r'saved\s+([\d,]+)\s+polygon[s]?', msg_lower, re.IGNORECASE)
            if poly_match:
                count_str = poly_match.group(1).replace(',', '')
                polygons_created = max(polygons_created, int(count_str))
            # Also try generic polygon count
            if polygons_created == 0:
                poly_match2 = re.search(r'(\d+)\s*polygon[s]?', msg_lower, re.IGNORECASE)
                if poly_match2:
                    polygons_created = max(polygons_created, int(poly_match2.group(1)))
            # Look for clusters: "K-means completed with {num} clusters" or "high_ndvi_clusters: {num}"
            cluster_match = re.search(r'(?:completed|using|with)\s+(\d+)\s+cluster[s]?', msg_lower, re.IGNORECASE)
            if cluster_match:
                clusters_used = max(clusters_used, int(cluster_match.group(1)))
            # Also try generic cluster count
            if clusters_used == 0:
                cluster_match2 = re.search(r'(\d+)\s*cluster[s]?', msg_lower, re.IGNORECASE)
                if cluster_match2:
                    clusters_used = max(clusters_used, int(cluster_match2.group(1)))
        
        quality_check["metrics"] = {
            "polygons_created": polygons_created,
            "clusters_used": clusters_used
        }
    
    elif step_id == "prediction" or step_id == "normal_prediction":
        # Look for prediction metrics
        predictions_made = 0
        confidence_avg = None
        class_percentages = {}
        
        # Get base output directory
        base_output = config.get("output", {}).get("base_output_dir") or config.get("base_output_dir", "")
        if not base_output:
            aoi = config.get("aoi", "")
            if aoi and "." in aoi:
                aoi_suffix = aoi.split(".")[-1]
            elif aoi:
                aoi_suffix = aoi
            else:
                aoi_suffix = "unknown"
            base_output = str(Path(LOCAL_OUTPUTS_ROOT) / aoi_suffix)
        
        try:
            base_output_path = Path(base_output)
            for snapshot in ["previous", "current"]:
                raw_dir = base_output_path / "time_series" / snapshot / "predictions" / "raw"
                gpkg_files = sorted(raw_dir.glob("*_predicted.gpkg")) if raw_dir.exists() else []
                if gpkg_files:
                    gdf = gpd.read_file(gpkg_files[0])
                    if 'predicted_class' in gdf.columns:
                        total = len(gdf[gdf['predicted_class'] >= 0])
                        if total > 0:
                            class_counts = gdf[gdf['predicted_class'] >= 0]['predicted_class'].value_counts()
                            class_percentages = {
                                'grassland_percent': round((class_counts.get(0, 0) / total) * 100, 2),
                                'tree_percent': round((class_counts.get(1, 0) / total) * 100, 2),
                                'urban_percent': round((class_counts.get(2, 0) / total) * 100, 2)
                            }
                            logger.info(f"Read class percentages from first file: {gpkg_files[0]}")
                            break
        except Exception as e:
            logger.warning(f"Failed to read local prediction GPKG: {e}")
        
        # Extract from logs
        for log in logs:
            msg = log.get("message", "")
            msg_lower = msg.lower()
            # Look for "Total predictions: {count}"
            pred_match = re.search(r'total\s+predictions?[:\s]+([\d,]+)', msg_lower, re.IGNORECASE)
            if pred_match:
                count_str = pred_match.group(1).replace(',', '')
                predictions_made = max(predictions_made, int(count_str))
            # Also try generic prediction count
            if predictions_made == 0:
                pred_match2 = re.search(r'(\d+)\s*prediction[s]?', msg_lower, re.IGNORECASE)
                if pred_match2:
                    predictions_made = max(predictions_made, int(pred_match2.group(1)))
            # Extract confidence
            conf_match = re.search(r'(?:average|avg|mean)\s+confidence[:\s]+([\d.]+)', msg_lower, re.IGNORECASE)
            if conf_match:
                confidence_avg = float(conf_match.group(1))
        
        quality_check["metrics"] = {
            "predictions_made": predictions_made,
            "average_confidence": confidence_avg,
            "class_percentages": class_percentages if class_percentages else None
        }
    
    elif step_id == "change_detection":
        # Look for change detection metrics
        tiles_analyzed = 0
        
        for log in logs:
            msg = log.get("message", "")
            msg_lower = msg.lower()
            # Look for "Found {count} tiles to process"
            tile_match = re.search(r'found\s+(\d+)\s+tiles?\s+to\s+process', msg_lower, re.IGNORECASE)
            if tile_match:
                tiles_analyzed = max(tiles_analyzed, int(tile_match.group(1)))
            # Also try generic tile count
            if tiles_analyzed == 0:
                tile_match2 = re.search(r'(\d+)\s+tiles?', msg_lower, re.IGNORECASE)
                if tile_match2:
                    tiles_analyzed = max(tiles_analyzed, int(tile_match2.group(1)))
        
        quality_check["metrics"] = {
            "tiles_analyzed": tiles_analyzed
        }
    
    elif step_id == "tree_clustering":
        # Look for clustering metrics
        clusters_found = 0
        num_clusters = 0
        cluster_pixel_counts = {}
        
        # Get base output directory
        base_output = config.get("output", {}).get("base_output_dir") or config.get("base_output_dir", "")
        if not base_output:
            aoi = config.get("aoi", "")
            if aoi and "." in aoi:
                aoi_suffix = aoi.split(".")[-1]
            elif aoi:
                aoi_suffix = aoi
            else:
                aoi_suffix = "unknown"
            base_output = str(Path(LOCAL_OUTPUTS_ROOT) / aoi_suffix)
        
        try:
            change_detection_dir = Path(base_output) / "change_detection"
            cluster_rasters = sorted(change_detection_dir.glob("tile_*/tree_clusters/*.tif")) + sorted(change_detection_dir.glob("tile_*/tree_clusters/*.tiff"))
            if cluster_rasters:
                with rasterio.open(cluster_rasters[0]) as src:
                    cluster_data = src.read(1)
                    nodata = src.nodata if src.nodata is not None else -9999
                    valid_mask = cluster_data != nodata
                    unique_clusters, counts = np.unique(cluster_data[valid_mask], return_counts=True)
                    cluster_pixel_counts = {
                        int(cluster_id): int(count)
                        for cluster_id, count in zip(unique_clusters, counts)
                        if cluster_id >= 0
                    }
                    clusters_found = len([c for c in unique_clusters if c >= 0])
                    num_clusters = clusters_found
                    logger.info(f"Read cluster pixel counts from first file: {cluster_rasters[0]}")
        except Exception as e:
            logger.warning(f"Failed to read local cluster raster: {e}")
        
        # Also extract from logs as fallback
        for log in logs:
            msg = log.get("message", "")
            msg_lower = msg.lower()
            # Look for "Found {n} clusters" (HDBSCAN)
            found_match = re.search(r'found\s+(\d+)\s+clusters?', msg_lower, re.IGNORECASE)
            if found_match:
                clusters_found = max(clusters_found, int(found_match.group(1)))
            # Look for "K-Means completed with {num} clusters"
            kmeans_match = re.search(r'(?:k-means|kmeans)\s+completed\s+with\s+(\d+)\s+clusters?', msg_lower, re.IGNORECASE)
            if kmeans_match:
                num_clusters = max(num_clusters, int(kmeans_match.group(1)))
                clusters_found = max(clusters_found, num_clusters)
            # Also try generic cluster count
            if clusters_found == 0:
                cluster_match = re.search(r'(\d+)\s+clusters?', msg_lower, re.IGNORECASE)
                if cluster_match:
                    clusters_found = max(clusters_found, int(cluster_match.group(1)))
        
        quality_check["metrics"] = {
            "clusters_found": clusters_found,
            "num_clusters": num_clusters if num_clusters > 0 else clusters_found,
            "cluster_pixel_counts": cluster_pixel_counts if cluster_pixel_counts else None
        }
    
    elif step_id == "vitality":
        # Count pixels directly from vitality_classes.tif raster file (not from summary.json)
        high_vitality_percent = None
        medium_vitality_percent = None
        medium_morbidity_percent = None
        severe_morbidity_percent = None
        no_change_percent = None
        total_pixels = None
        pixel_counts = {}
        
        # Try to read vitality_classes.tif raster from S3
        base_output = config.get("output", {}).get("base_output_dir") or config.get("base_output_dir", "")
        if not base_output:
            aoi = config.get("aoi", "")
            if aoi and "." in aoi:
                aoi_suffix = aoi.split(".")[-1]
            elif aoi:
                aoi_suffix = aoi
            else:
                aoi_suffix = "unknown"
            base_output = str(Path(LOCAL_OUTPUTS_ROOT) / aoi_suffix)
        
        try:
            raster_files = sorted(Path(base_output).glob("vitality_autoencoder/tile_*/vitality_classes.tif"))
            if raster_files:
                first_raster_path = raster_files[0]
                with rasterio.open(first_raster_path) as src:
                    raster_data = src.read(1)
                    nodata = src.nodata if src.nodata is not None else 255
                    valid_mask = raster_data != nodata
                    unique_classes, counts = np.unique(raster_data[valid_mask], return_counts=True)
                    class_mapping = {
                        0: "severe_morbidity",
                        1: "medium_morbidity",
                        2: "medium_vitality",
                        3: "high_vitality",
                        4: "no_change"
                    }
                    total_valid_pixels = valid_mask.sum()
                    total_pixels = int(total_valid_pixels)
                    for class_id, count in zip(unique_classes, counts):
                        class_name = class_mapping.get(int(class_id))
                        if class_name:
                            pixel_counts[class_name] = int(count)
                    if total_valid_pixels > 0:
                        high_vitality_percent = round((pixel_counts.get("high_vitality", 0) / total_valid_pixels) * 100, 2)
                        medium_vitality_percent = round((pixel_counts.get("medium_vitality", 0) / total_valid_pixels) * 100, 2)
                        medium_morbidity_percent = round((pixel_counts.get("medium_morbidity", 0) / total_valid_pixels) * 100, 2)
                        severe_morbidity_percent = round((pixel_counts.get("severe_morbidity", 0) / total_valid_pixels) * 100, 2)
                        no_change_percent = round((pixel_counts.get("no_change", 0) / total_valid_pixels) * 100, 2)
                        logger.info(f"Counted pixels from raster {first_raster_path}:")
                        logger.info(f"  Total valid pixels: {total_valid_pixels:,}")
                        logger.info(f"  Pixel counts: {pixel_counts}")
                        logger.info(f"  Percentages: high_vitality={high_vitality_percent}%, medium_vitality={medium_vitality_percent}%, medium_morbidity={medium_morbidity_percent}%, severe={severe_morbidity_percent}%, no_change={no_change_percent}%")
        except Exception as e:
            logger.warning(f"Failed to read local vitality raster: {e}")
        
        # Collect warnings for quality check
        quality_warnings = []
        if medium_morbidity_percent is not None:
            medium_morbidity_percentile = config.get("vitality", {}).get("medium_morbidity_percentile", 0.85)
            severe_morbidity_percentile = config.get("vitality", {}).get("severe_morbidity_percentile", 0.95)
            expected_medium_morbidity = (severe_morbidity_percentile - medium_morbidity_percentile) * 100
            medium_morbidity_tolerance = 3.0
            if abs(medium_morbidity_percent - expected_medium_morbidity) > medium_morbidity_tolerance:
                quality_warnings.append(
                    f"Medium Morbidity: {medium_morbidity_percent:.2f}% (expected ~{expected_medium_morbidity:.1f}%)"
                )
        if no_change_percent is not None and no_change_percent == 0.0:
            no_change_ndvi = config.get("vitality", {}).get("no_change_abs_delta_ndvi", 0.03)
            no_change_ndmi = config.get("vitality", {}).get("no_change_abs_delta_ndmi", 0.03)
            if no_change_ndvi is not None or no_change_ndmi is not None:
                quality_warnings.append(
                    f"No Change: {no_change_percent:.2f}% (expected > 0% with thresholds set)"
                )
        
        quality_check["metrics"] = {
            "high_vitality_percent": high_vitality_percent,
            "medium_vitality_percent": medium_vitality_percent,
            "medium_morbidity_percent": medium_morbidity_percent,
            "severe_morbidity_percent": severe_morbidity_percent,
            "no_change_percent": no_change_percent,
            "total_pixels": total_pixels,
            "pixel_counts": pixel_counts if pixel_counts else None,
            "warnings": quality_warnings if quality_warnings else None  # Add warnings to metrics
        }
    
    # Add log summary (last 10 important log entries)
    important_logs = [log.get("message", "") for log in logs[-10:] if any(keyword in log.get("message", "").lower() for keyword in ["completed", "finished", "saved", "created", "processed", "downloaded", "error", "warning"])]
    quality_check["logs_summary"] = important_logs
    
    # Save to local outputs
    aoi = config.get("aoi", "")
    if aoi and "." in aoi:
        aoi_suffix = aoi.split(".")[-1]
    elif aoi:
        aoi_suffix = aoi
    else:
        aoi_suffix = "unknown"
    
    base_output = str(Path(LOCAL_OUTPUTS_ROOT) / aoi_suffix)
    quality_check_path = str(Path(base_output) / "quality_checks" / f"{step_id}_quality_check.json")
    
    try:
        local_path = Path(quality_check_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "w") as f:
            json_lib.dump(quality_check, f, indent=2)
        quality_check["local_path"] = str(local_path)
    except Exception as e:
        logger.warning(f"Failed to save quality check to {quality_check_path}: {e}")
        quality_check["save_error"] = str(e)
    
    return quality_check


def collect_results(step_id, config):
    """Collect results and a human-readable summary for a completed step."""
    results = {
        "step_id": step_id,
        "outputs": [],
        "visualizations": [],
        "summary": "",
        "quality_check": None
    }
    
    # Derive base_output from AOI name
    aoi = config.get("aoi", "")
    if aoi and "." in aoi:
        aoi_suffix = aoi.split(".")[-1]
    elif aoi:
        aoi_suffix = aoi
    else:
        aoi_suffix = "bayernwerk_kmeans_100km"  # fallback
    base_output = str(Path(LOCAL_OUTPUTS_ROOT) / aoi_suffix)
    
    # Basic per-step summaries and key output locations
    date_str = config.get("date", "N/A")
    window_days = config.get("window_days", 45)
    tree_mask_source = config.get("tree_mask_source", "previous")
    snapshots = config.get("snapshots", "both")

    # Time-series workflow steps
    if step_id == "download":
        results["summary"] = (
            f"Downloaded Sentinel-2 imagery for AOI '{aoi_suffix}' around {date_str} "
            f"with a ±{window_days//2} day window for both current and previous snapshots. "
            f"Raw bands are stored under '{base_output}/time_series/<current|previous>/raw_data'."
        )
        results["outputs"] = [
            f"{base_output}/time_series/current/raw_data",
            f"{base_output}/time_series/previous/raw_data",
        ]
    elif step_id == "preprocess":
        results["summary"] = (
            f"Built cloud-masked, co-registered band stacks for AOI '{aoi_suffix}' "
            f"for both time steps, ready for segmentation and prediction. "
            f"Processed rasters are stored under '{base_output}/time_series/<current|previous>/processed_data'."
        )
        results["outputs"] = [
            f"{base_output}/time_series/current/processed_data",
            f"{base_output}/time_series/previous/processed_data",
        ]
    elif step_id == "segmentation":
        results["summary"] = (
            f"Ran k-means segmentation and polygonization on the current snapshot for AOI '{aoi_suffix}', "
            f"producing vegetation polygons and zonal statistics for downstream tree prediction. "
            f"Segmentation outputs are stored under '{base_output}/time_series/previous/segmentation'."
        )
        results["outputs"] = [
            f"{base_output}/time_series/previous/segmentation",
        ]
    elif step_id == "prediction":
        results["summary"] = (
            f"Applied the tree classification model to zonal statistics for AOI '{aoi_suffix}', "
            f"producing per-polygon class predictions, confidence layers and rasters. "
            f"Predictions are stored under '{base_output}/time_series/previous/predictions'."
        )
        results["outputs"] = [
            f"{base_output}/time_series/previous/predictions",
        ]
    elif step_id == "change_detection":
        results["summary"] = (
            f"Organized current and previous imagery and predictions for AOI '{aoi_suffix}' into a unified "
            f"change-detection layout, computing NDVI/NDMI time-step pairs per tile. "
            f"Outputs are stored under '{base_output}/change_detection'."
        )
        results["outputs"] = [
            f"{base_output}/change_detection",
        ]
    elif step_id == "tree_clustering":
        results["summary"] = (
            f"Ran PCA + HDBSCAN tree clustering over masked change-detection rasters for AOI '{aoi_suffix}', "
            f"identifying tree communities and writing per-tile cluster rasters. "
            f"Clustered rasters are stored under '{base_output}/change_detection/tile_*/tree_clusters'."
        )
        results["outputs"] = [
            f"{base_output}/change_detection/tile_*/tree_clusters",
        ]
    elif step_id == "vitality":
        results["summary"] = (
            f"Trained per-cluster autoencoders on tree pixels and scored vitality/morbidity for AOI '{aoi_suffix}', "
            f"using tree_mask_source='{tree_mask_source}' and snapshots='{snapshots}'. "
            f"Vitality rasters and summaries are stored under '{base_output}/vitality_autoencoder/tile_*'."
        )
        results["outputs"] = [
            f"{base_output}/vitality_autoencoder/tile_*",
        ]

    # Classic single-date (normal) workflow steps
    elif step_id == "normal_download":
        results["summary"] = (
            f"Downloaded Sentinel-2 imagery for AOI '{aoi_suffix}' on {date_str} "
            f"for the single-date (vegbins) workflow. "
            f"Raw bands are stored under '{base_output}/raw_data'."
        )
        results["outputs"] = [f"{base_output}/raw_data"]
    elif step_id == "normal_asset_prep":
        results["summary"] = (
            f"Prepared AOI assets and internal tiling for AOI '{aoi_suffix}', "
            f"including geometry normalization and tile index creation. "
            f"Assets are stored under '{base_output}/asset_preparation'."
        )
        results["outputs"] = [f"{base_output}/asset_preparation"]
    elif step_id == "normal_process_imagery":
        results["summary"] = (
            f"Processed and tiled Sentinel-2 bands for AOI '{aoi_suffix}' for the single-date workflow, "
            f"producing analysis-ready rasters under '{base_output}/processed_data'."
        )
        results["outputs"] = [f"{base_output}/processed_data"]
    elif step_id == "normal_segmentation":
        results["summary"] = (
            f"Segmented the scene for AOI '{aoi_suffix}' using NDVI-based k-means, "
            f"creating polygons and zonal stats for tree prediction. "
            f"Segmentation outputs are stored under '{base_output}/segmentation'."
        )
        results["outputs"] = [f"{base_output}/segmentation"]
    elif step_id == "normal_prediction":
        results["summary"] = (
            f"Ran the tree prediction model for AOI '{aoi_suffix}' in single-date mode, "
            f"producing classification polygons and rasters under '{base_output}/predictions'."
        )
        results["outputs"] = [f"{base_output}/predictions"]
    elif step_id == "normal_folium":
        results["summary"] = (
            f"Built an interactive Folium map for AOI '{aoi_suffix}', "
            f"linking to key prediction and segmentation layers for quick QA."
        )
        results["outputs"] = [f"{base_output}/folium"]

    # Fallback: generic placeholder
    if not results["outputs"]:
        results["outputs"] = [f"{base_output}/{step_id}"]
    if not results["summary"]:
        results["summary"] = (
            f"Step '{step_id}' completed for AOI '{aoi_suffix}'. "
            f"Outputs are available under '{base_output}/{step_id}'."
        )
    
    return results


@app.route('/')
def index():
    """Serve the main frontend page."""
    return send_from_directory('static', 'index.html')

@app.route('/api/port', methods=['GET'])
def get_port():
    """Get the current server port for frontend auto-detection."""
    return jsonify({"port": request.environ.get('SERVER_PORT', '5000')})


@app.route('/api/models', methods=['GET'])
def list_models():
    """List all local model checkpoint files from the model_checkpoints directory."""
    try:
        models_dir = Path(__file__).parent.parent / "model_checkpoints"
        models = []
        if models_dir.exists():
            for path in models_dir.rglob("*.pt"):
                stat = path.stat()
                models.append({
                    "filename": path.name,
                    "full_path": str(path),
                    "size": stat.st_size,
                    "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Sort by last modified (newest first)
        models.sort(key=lambda x: x.get("last_modified", ""), reverse=True)
        
        return jsonify({"models": models})
    except Exception as e:
        logger.error(f"Error listing local models: {e}")
        return jsonify({"error": str(e), "models": []}), 500


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Treelance Web Frontend')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (NEVER use in production)')
    args = parser.parse_args()
    
    port = args.port
    
    # Security: Debug mode is controlled by environment variable or explicit flag
    # NEVER enable debug mode in production - it exposes stack traces and enables code execution
    debug_enabled = DEBUG_MODE or args.debug
    
    if debug_enabled:
        logger.warning("⚠️  DEBUG MODE ENABLED - Do NOT use in production!")
        print("⚠️  WARNING: Debug mode is enabled. This should NEVER be used in production.")
    
    print(f"🌐 Server starting on http://{args.host}:{port}")
    print(f"🔒 Debug mode: {'ENABLED (unsafe)' if debug_enabled else 'DISABLED (safe)'}")
    print(f"🔒 CORS origins: {ALLOWED_ORIGINS}")
    
    app.run(host=args.host, port=port, debug=debug_enabled, use_reloader=False)
