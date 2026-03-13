from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import requests
import geopandas as gpd
from loguru import logger
from pystac_client import Client
from shapely.geometry import shape, mapping
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
from shapely.wkt import loads as load_wkt
import rasterio
from rasterio.io import DatasetReader
from rasterio.errors import RasterioIOError
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_geom
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
import numpy as np
from datetime import datetime, date

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    gdal = None


EARTHSEARCH_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"


# =========================
# User variables (edit here)
# =========================
# Date range (either as strings YYYY-MM-DD or computed from below)
USER_START_DATE = "2025-01-01"
USER_END_DATE = "2025-01-06"
USER_TIME_RANGE = f"{USER_START_DATE}/{USER_END_DATE}"
USER_AOI = "data/aoi/aoi_big.geojson"
USER_OUTPUT_DIR = "local_output/imagery"
USER_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
USER_MAX_CLOUD_COVER = 10
USER_PICK_LATEST = False
USER_KEEP_INDIVIDUAL_BANDS = False
USER_CLIP_TO_AOI = False
USER_STAC_QUERY: Dict = {}


# Asset key aliases between band codes and EarthSearch asset names
BAND_ALIAS: Dict[str, str] = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
    "AOT": "aot",
    "SCL": "scl",
    "WVP": "wvp",
    "TCI": "visual",
}

REVERSE_BAND_ALIAS: Dict[str, str] = {v: k for k, v in BAND_ALIAS.items()}


def _ensure_output_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dedupe_ring(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Remove duplicate consecutive coordinates and ensure closed ring with >=4 points."""
    if not coords:
        raise ValueError("Empty ring coordinates")
    deduped: List[Tuple[float, float]] = []
    prev_x: Optional[float] = None
    prev_y: Optional[float] = None
    for x, y in coords:
        if prev_x is None or prev_y is None or (x != prev_x or y != prev_y):
            deduped.append((x, y))
        prev_x, prev_y = x, y
    if len(deduped) >= 1 and deduped[0] != deduped[-1]:
        deduped.append(deduped[0])
    if len(deduped) < 4:
        raise ValueError("AOI ring has too few unique points after cleaning")
    return deduped


def _clean_geometry_dict(geom: Dict) -> Dict:
    """Clean polygon/multipolygon geometry by removing duplicate consecutive vertices."""
    g = shape(geom)
    if g.is_empty:
        raise ValueError("AOI geometry is empty")

    if g.geom_type == "Polygon":
        exterior = _dedupe_ring(list(g.exterior.coords))
        interiors = [
            _dedupe_ring(list(r.coords)) for r in g.interiors if len(set(r.coords)) > 2
        ]
        return mapping(Polygon(exterior, interiors))

    if g.geom_type == "MultiPolygon":
        cleaned_polys = []
        for poly in g.geoms:
            exterior = _dedupe_ring(list(poly.exterior.coords))
            interiors = [
                _dedupe_ring(list(r.coords))
                for r in poly.interiors
                if len(set(r.coords)) > 2
            ]
            cleaned_polys.append(Polygon(exterior, interiors))
        # Keep as MultiPolygon
        return mapping(unary_union(cleaned_polys))

    # For non-area geometries, fallback to original
    return mapping(g)


def _extract_mgrs_tile(item_id: str) -> str:
    """
    Extract MGRS tile identifier from Sentinel-2 item ID.
    
    Example: S2A_33UWP_20250501_1_L2A -> 33UWP
    
    Args:
        item_id: Sentinel-2 item ID
    
    Returns:
        MGRS tile identifier (e.g., "33UWP")
    """
    # Sentinel-2 item ID format: S2A_33UWP_20250501_1_L2A
    # MGRS tile is the second part (index 1) after splitting by '_'
    parts = item_id.split("_")
    if len(parts) >= 2:
        return parts[1]
    raise ValueError(f"Could not extract MGRS tile from item ID: {item_id}")


def _load_geometry(aoi: Union[str, Dict, Path]) -> Dict:
    """
    Load AOI geometry and return a GeoJSON-like dict geometry.

    Accepts one of:
    - Database table name (format: "schema.table") - loads from liveeoshps database
    - Path to GeoJSON file (Feature or FeatureCollection)
    - S3 path to GeoJSON file (s3://bucket/key)
    - WKT string (Polygon/MultiPolygon)
    - GeoJSON geometry/feature dict
    """
    if isinstance(aoi, str) and aoi.startswith("s3://"):
        raise ValueError("S3 AOI paths are no longer supported. Please provide a local AOI file path.")

    if isinstance(aoi, (str, Path)) and Path(str(aoi)).exists():
        with open(aoi, "r", encoding="utf-8") as f:
            data = json.load(f)
        # If FeatureCollection, union all features' geometries
        if data.get("type") == "FeatureCollection":
            if not data.get("features"):
                raise ValueError("Empty FeatureCollection provided as AOI")
            geoms = [shape(feat["geometry"]) for feat in data["features"]]
            unioned = unary_union(geoms)
            geom = mapping(unioned)
        elif data.get("type") == "Feature":
            geom = data["geometry"]
        else:
            # assume Geometry dict
            geom = data
        # Clean and validate geometry
        cleaned = _clean_geometry_dict(geom)
        _ = shape(cleaned)
        return cleaned

    if isinstance(aoi, str):
        if "." in aoi and not any(char in aoi for char in ["/", "\\", " "]):
            raise ValueError("Database-style AOI identifiers are no longer supported. Please provide a local AOI file path or WKT.")
        # try WKT
        try:
            g = load_wkt(aoi)
            if not isinstance(g, BaseGeometry):
                raise ValueError("Provided WKT did not parse to a geometry")
            return json.loads(json.dumps(g.__geo_interface__))
        except Exception as e:
            raise ValueError("AOI string must be a valid file path or WKT") from e

    if isinstance(aoi, dict):
        geom = aoi if aoi.get("type") != "Feature" else aoi["geometry"]
        cleaned = _clean_geometry_dict(geom)
        _ = shape(cleaned)
        return cleaned

    raise ValueError("Unsupported AOI input")


def _http_href_from_s3(href: str) -> str:
    """
    Convert an s3://sentinel-cogs path to its equivalent HTTPS URL.
    If href already HTTP(S), return unchanged.
    """
    if href.startswith("s3://sentinel-cogs"):
        return href.replace(
            "s3://sentinel-cogs", "https://sentinel-cogs.s3.us-west-2.amazonaws.com", 1
        )
    return href


def _s3_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _download_file(url: str, dst_path: Path, chunk_size: int = 8 * 1024 * 1024) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return dst_path


def _generate_imagery_summary_report(
    tile_results: List[Dict[str, Any]], 
    output_dir: Union[str, Path], 
    bands: List[str]
) -> None:
    """
    Generate a summary report of all downloaded imagery with quality metadata.
    
    Args:
        tile_results: List of tile result dictionaries with metadata
        output_dir: Output directory (local or S3) where report will be saved
        bands: List of bands that were downloaded
    """
    import json
    import csv
    from datetime import datetime as dt
    
    if not tile_results:
        logger.warning("No tile results to generate summary report")
        return
    
    # Prepare summary data
    summary_data = {
        "generated_at": dt.now().isoformat(),
        "total_scenes": len(tile_results),
        "bands": bands,
        "scenes": []
    }
    
    # Calculate statistics
    cloud_covers = [tr.get("cloud_cover") for tr in tile_results if tr.get("cloud_cover") is not None]
    aoi_coverages = [tr.get("aoi_coverage_pct") for tr in tile_results if tr.get("aoi_coverage_pct") is not None]
    
    if cloud_covers:
        summary_data["cloud_cover_stats"] = {
            "min": float(min(cloud_covers)),
            "max": float(max(cloud_covers)),
            "mean": float(sum(cloud_covers) / len(cloud_covers)),
            "median": float(sorted(cloud_covers)[len(cloud_covers) // 2])
        }
    
    if aoi_coverages:
        summary_data["aoi_coverage_stats"] = {
            "min": float(min(aoi_coverages)),
            "max": float(max(aoi_coverages)),
            "mean": float(sum(aoi_coverages) / len(aoi_coverages)),
            "total_coverage": float(sum(aoi_coverages))
        }
    
    # Add scene details
    for tr in tile_results:
        scene_info = {
            "tile_id": tr.get("tile_id"),
            "item_id": tr.get("item_id"),
            "stacked_path": tr.get("stacked_path"),
            "cloud_cover": tr.get("cloud_cover"),
            "datetime": tr.get("datetime"),
            "aoi_coverage_pct": tr.get("aoi_coverage_pct"),
        }
        summary_data["scenes"].append(scene_info)
    
    # Determine output path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_path = output_path / "imagery_quality_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"📊 Imagery quality summary (JSON) saved to: {json_path}")
    
    # Save CSV report
    csv_path = output_path / "imagery_quality_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tile_id", "item_id", "cloud_cover", "datetime", "aoi_coverage_pct", "stacked_path"
        ])
        for tr in tile_results:
            writer.writerow([
                tr.get("tile_id", ""),
                tr.get("item_id", ""),
                tr.get("cloud_cover", ""),
                tr.get("datetime", ""),
                tr.get("aoi_coverage_pct", ""),
                tr.get("stacked_path", "")
            ])
    logger.info(f"📊 Imagery quality summary (CSV) saved to: {csv_path}")


def _apply_cloud_mask_to_stack(stacked_path: Path, scl_path: Union[str, Path]) -> None:
    """
    Apply cloud and water mask from SCL (Scene Classification Layer) band to stacked imagery.
    
    SCL values to mask out (set to nodata):
    - 3: cloud shadows
    - 6: water (rivers, lakes, etc.)
    - 8: cloud medium probability
    - 9: cloud high probability
    - 10: thin cirrus
    
    Args:
        stacked_path: Path to stacked multi-band GeoTIFF
        scl_path: Path to SCL band GeoTIFF
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    
    logger.info(f"Creating cloud and water mask from SCL band: {scl_path}")
    
    with rasterio.open(stacked_path, "r+") as stacked_ds:
        with rasterio.open(scl_path) as scl_ds:
            # Read SCL band
            scl_data = scl_ds.read(1)
            
            # Ensure SCL matches stacked raster dimensions
            if (scl_ds.width != stacked_ds.width or 
                scl_ds.height != stacked_ds.height or
                scl_ds.transform != stacked_ds.transform or
                scl_ds.crs != stacked_ds.crs):
                logger.info("Reprojecting SCL band to match stacked raster dimensions...")
                scl_reproj = np.zeros((stacked_ds.height, stacked_ds.width), dtype=scl_data.dtype)
                reproject(
                    source=scl_data,
                    destination=scl_reproj,
                    src_transform=scl_ds.transform,
                    src_crs=scl_ds.crs,
                    dst_transform=stacked_ds.transform,
                    dst_crs=stacked_ds.crs,
                    resampling=Resampling.nearest
                )
                scl_data = scl_reproj
            
            # Create mask: True where clouds or water are present (values 3, 6, 8, 9, 10)
            mask = np.isin(scl_data, [3, 6, 8, 9, 10])
            n_masked_pixels = int(mask.sum())
            total_pixels = mask.size
            masked_percentage = (n_masked_pixels / total_pixels * 100) if total_pixels > 0 else 0
            
            # Count cloud vs water pixels separately for logging
            cloud_mask = np.isin(scl_data, [3, 8, 9, 10])
            water_mask = (scl_data == 6)
            n_cloud_pixels = int(cloud_mask.sum())
            n_water_pixels = int(water_mask.sum())
            
            logger.info(
                f"SCL mask: {n_masked_pixels:,} pixels masked ({masked_percentage:.2f}% of image) - "
                f"Clouds: {n_cloud_pixels:,}, Water: {n_water_pixels:,}"
            )
            
            # Get nodata value from stacked raster
            nodata_value = stacked_ds.nodata if stacked_ds.nodata is not None else 0
            
            # Apply mask to all bands
            for band_idx in range(1, stacked_ds.count + 1):
                band_data = stacked_ds.read(band_idx)
                # Set cloud and water pixels to nodata
                band_data[mask] = nodata_value
                stacked_ds.write(band_data, band_idx)
            
            logger.info(f"✅ Cloud and water mask applied to {stacked_ds.count} bands. {n_masked_pixels:,} pixels masked.")


def _stack_bands_to_multitiff(band_paths: List[Path], out_path: Path) -> Path:
    """Stack multiple single-band rasters into a multi-band GeoTIFF.
    
    This function safely opens multiple rasterio datasets using ExitStack,
    ensuring all datasets are properly closed even if an error occurs
    during the opening of any one of them.
    
    Args:
        band_paths: List of paths to single-band raster files
        out_path: Output path for the stacked multi-band GeoTIFF
        
    Returns:
        Path to the output file
        
    Raises:
        ValueError: If no band paths are provided
        rasterio.errors.RasterioIOError: If a band file cannot be opened
    """
    from contextlib import ExitStack
    
    if not band_paths:
        raise ValueError("No band paths provided to stack")

    # Use ExitStack to safely manage multiple datasets
    # This ensures all opened datasets are closed even if opening one fails
    with ExitStack() as stack:
        datasets: List[DatasetReader] = []
        for p in band_paths:
            try:
                ds = stack.enter_context(rasterio.open(str(p)))
                datasets.append(ds)
            except Exception as e:
                # Log which file failed to open
                logger.error(f"Failed to open band file {p}: {e}")
                raise
        ref = datasets[0]
        width, height = ref.width, ref.height
        transform: Affine = ref.transform
        crs = ref.crs

        # Step A: Resampling - Co-register and resample bands to match reference resolution
        # 20m bands (B05, B11, B12) are upscaled to 10m using Bilinear Interpolation
        # to match 10m bands (B02, B03, B04, B08)
        # This ensures all bands are pixel-aligned for clustering and model input
        prepared_arrays: List = []
        for idx, ds in enumerate(datasets):
            if (
                ds.width == width
                and ds.height == height
                and ds.transform == transform
                and ds.crs == crs
            ):
                # Band already matches reference grid (same resolution)
                prepared_arrays.append(ds.read(1))
            else:
                # Resample band to match reference grid using bilinear interpolation
                # This handles 20m -> 10m upscaling for B05, B11, B12
                src = ds.read(1)
                dst = np.empty((height, width), dtype=src.dtype)
                reproject(
                    source=src,
                    destination=dst,
                    src_transform=ds.transform,
                    src_crs=ds.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear,  # Bilinear interpolation for smooth upscaling
                )
                prepared_arrays.append(dst)

        meta = ref.meta.copy()
        meta.update(
            {
                "count": len(datasets),
                "driver": "GTiff",
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "BIGTIFF": "IF_SAFER",
            }
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            for band_index, arr in enumerate(prepared_arrays, start=1):
                dst.write(arr, indexes=band_index)

            # Assign human-readable band names based on band codes present in filenames
            for band_index, p in enumerate(band_paths, start=1):
                try:
                    band_code = p.stem.split("_")[-1]
                    band_alias = BAND_ALIAS.get(band_code, band_code)
                    # e.g., "B02 - blue"
                    dst.set_band_description(band_index, f"{band_code} - {band_alias}")
                except Exception:
                    # Description setting is best-effort; continue if not supported
                    pass

        return out_path
        # ExitStack automatically closes all datasets when exiting the with block


@dataclass
@dataclass
class Sentinel2DownloadResult:
    item_id: str  # Primary item ID (for backward compatibility)
    band_files: Dict[str, str]
    stacked_path: Optional[str]  # Primary stacked path (for backward compatibility)
    tiles_fc_path: Optional[str] = None
    tiles_merged_path: Optional[str] = None
    # New fields for multi-tile support
    tile_results: Optional[List[Dict[str, str]]] = None  # List of {tile_id, item_id, stacked_path} per tile


def download_sentinel2(
    time: str,
    aoi: Union[str, Dict, Path],
    output_dir: Union[str, Path],
    bands: Optional[List[str]] = None,
    max_cloud_cover: int = 15,
    pick_latest: bool = True,
    download_all_tiles: bool = False,
    dry_run: bool = False,
    additional_query: Optional[Dict] = None,
    keep_individual_bands: bool = False,
    clip_to_aoi: bool = False,
    buffer_distance: Optional[float] = None,
) -> Sentinel2DownloadResult:
    """
    Search, download selected Sentinel-2 L2A bands, and stack them into one GeoTIFF.

    Args:
        time: ISO8601 datetime or range (e.g., "2024-06-10" or "2024-06-01/2024-06-30").
        aoi: AOI as file path to GeoJSON, WKT string, or GeoJSON-like dict.
        output_dir: Directory where downloads and the stacked file will be written.
        bands: Asset keys to download and stack, default 10m set ["B02","B03","B04","B08"].
        max_cloud_cover: Maximum eo:cloud_cover percentage filter.
        pick_latest: If multiple items found, pick the most recent (else least cloud cover).
        buffer_distance: Optional buffer distance in meters to apply to AOI before searching.

    Returns:
        Sentinel2DownloadResult with item_id, per-band paths, and stacked path.
    """
    bands = bands or ["B02", "B03", "B04", "B08"]
    geom = _load_geometry(aoi)
    
    # Apply buffer if specified
    if buffer_distance is not None and buffer_distance > 0:
        logger.info(f"Applying {buffer_distance}m buffer to AOI before Sentinel search")
        # Convert GeoJSON dict to GeoDataFrame for buffering
        geom_gdf = gpd.GeoDataFrame.from_features([{
            "type": "Feature",
            "properties": {},
            "geometry": geom
        }], crs="EPSG:4326")
        
        # Buffer in Web Mercator (for accurate distance in meters)
        geom_gdf_projected = geom_gdf.to_crs(epsg=3857)
        geom_gdf_projected.geometry = geom_gdf_projected.geometry.buffer(buffer_distance)
        geom_gdf_buffered = geom_gdf_projected.to_crs("EPSG:4326")
        
        # Simplify geometry to reduce vertex count for STAC API (prevents "request entity too large" errors)
        # Tolerance of 0.0001 degrees ≈ 10 meters at equator - small enough to preserve shape
        original_geom = geom_gdf_buffered.geometry.iloc[0]
        # Count vertices before simplification
        def count_vertices(g):
            if hasattr(g, 'geoms'):  # MultiPolygon or MultiLineString
                return sum(count_vertices(subgeom) for subgeom in g.geoms)
            elif hasattr(g, 'exterior'):  # Polygon
                return len(g.exterior.coords) + sum(len(hole.coords) for hole in g.interiors)
            elif hasattr(g, 'coords'):  # LineString or Point
                return len(g.coords)
            return 0
        
        original_vertex_count = count_vertices(original_geom)
        geom_gdf_buffered.geometry = geom_gdf_buffered.geometry.simplify(tolerance=0.0001, preserve_topology=True)
        simplified_vertex_count = count_vertices(geom_gdf_buffered.geometry.iloc[0])
        
        if original_vertex_count > simplified_vertex_count:
            reduction_pct = ((original_vertex_count - simplified_vertex_count) / original_vertex_count) * 100
            logger.info(f"Simplified geometry: {original_vertex_count} → {simplified_vertex_count} vertices ({reduction_pct:.1f}% reduction)")
        
        # Convert back to GeoJSON dict
        buffered_geom = mapping(geom_gdf_buffered.geometry.iloc[0])
        geom = _clean_geometry_dict(buffered_geom)
        logger.info(f"✅ AOI buffered by {buffer_distance}m")
    else:
        # Even without buffering, simplify geometry if it's too complex for STAC API
        # Convert to GeoDataFrame for simplification
        geom_gdf = gpd.GeoDataFrame.from_features([{
            "type": "Feature",
            "properties": {},
            "geometry": geom
        }], crs="EPSG:4326")
        
        # Count vertices before simplification
        def count_vertices(g):
            if hasattr(g, 'geoms'):  # MultiPolygon or MultiLineString
                return sum(count_vertices(subgeom) for subgeom in g.geoms)
            elif hasattr(g, 'exterior'):  # Polygon
                return len(g.exterior.coords) + sum(len(hole.coords) for hole in g.interiors)
            elif hasattr(g, 'coords'):  # LineString or Point
                return len(g.coords)
            return 0
        
        original_vertex_count = count_vertices(geom_gdf.geometry.iloc[0])
        
        # Only simplify if geometry has many vertices (>1000) to avoid unnecessary processing
        if original_vertex_count > 1000:
            geom_gdf.geometry = geom_gdf.geometry.simplify(tolerance=0.0001, preserve_topology=True)
            simplified_vertex_count = count_vertices(geom_gdf.geometry.iloc[0])
            if simplified_vertex_count < original_vertex_count:
                reduction_pct = ((original_vertex_count - simplified_vertex_count) / original_vertex_count) * 100
                logger.info(f"Simplified geometry: {original_vertex_count} → {simplified_vertex_count} vertices ({reduction_pct:.1f}% reduction)")
                # Convert back to GeoJSON dict
                simplified_geom = mapping(geom_gdf.geometry.iloc[0])
                geom = _clean_geometry_dict(simplified_geom)

    out_dir = _ensure_output_dir(output_dir)

    logger.info("Querying Earth Search for Sentinel-2 L2A items")
    client = Client.open(EARTHSEARCH_URL)
    # Build STAC query: start from cloud cover, then merge any user-provided filters
    query_filters: Dict = {"eo:cloud_cover": {"lt": max_cloud_cover}}
    if additional_query:
        # shallow merge; user filters override defaults on same keys
        query_filters.update(additional_query)
    search = client.search(
        collections=[S2_COLLECTION],
        datetime=time,
        intersects=geom,
        query=query_filters,
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            "No Sentinel-2 L2A items found for given time/AOI constraints"
        )

    # Log available items with cloud cover and AOI coverage
    try:
        aoi_geom = shape(geom)
        aoi_area = aoi_geom.area if aoi_geom.area > 0 else None
    except Exception:
        aoi_geom = None
        aoi_area = None

    logger.info(f"Found {len(items)} items in time range. Listing availability:")
    # Sort for reporting: increasing cloud cover, then newest first
    def _cloud_cover_val(it):
        cc = it.properties.get("eo:cloud_cover")
        return float(cc) if cc is not None else 1e9

    reported = []
    for it in sorted(items, key=lambda x: (_cloud_cover_val(x), x.properties.get("datetime", "")), reverse=False):
        cc = it.properties.get("eo:cloud_cover")
        dt = it.properties.get("datetime") or it.properties.get("start_datetime") or ""
        cov_pct = None
        if aoi_geom is not None and it.geometry is not None and aoi_area:
            try:
                it_geom = shape(it.geometry)
                inter_area = it_geom.intersection(aoi_geom).area
                cov_pct = (inter_area / aoi_area) * 100 if aoi_area else None
            except Exception:
                cov_pct = None
        reported.append((it.id, dt, cc, cov_pct))

    for rid, dt, cc, cov in reported:
        if cov is None:
            logger.info(f"- {rid} | datetime={dt} | cloud_cover={cc}")
        else:
            logger.info(f"- {rid} | datetime={dt} | cloud_cover={cc} | aoi_coverage={cov:.2f}%")

    # Build tiles FeatureCollection and merged footprint for all candidate items
    tiles_fc_path: Optional[Path] = None
    tiles_merged_path: Optional[Path] = None
    try:
        features = []
        geoms = []
        for it in items:
            if it.geometry is None:
                continue
            g = shape(it.geometry)
            if aoi_geom is not None:
                try:
                    g = g.intersection(aoi_geom)
                except Exception:
                    pass
            if g.is_empty:
                continue
            geoms.append(g)
            props = {
                "id": it.id,
                "datetime": it.properties.get("datetime") or it.properties.get("start_datetime"),
                "eo:cloud_cover": it.properties.get("eo:cloud_cover"),
            }
            features.append({"type": "Feature", "geometry": mapping(g), "properties": props})

        if features:
            safe_time = str(time).replace("/", "_").replace(":", "-")
            tiles_fc_path = out_dir / f"tiles_{safe_time}.geojson"
            with open(tiles_fc_path, "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": features}, f)
            logger.info(f"Wrote tiles FeatureCollection: {tiles_fc_path}")

            try:
                merged = unary_union(geoms)
                tiles_merged_path = out_dir / f"tiles_merged_{safe_time}.geojson"
                with open(tiles_merged_path, "w", encoding="utf-8") as f:
                    json.dump({"type": "Feature", "geometry": mapping(merged), "properties": {}}, f)
                logger.info(f"Wrote merged tiles geometry: {tiles_merged_path}")
                
                # Calculate combined coverage and missing areas
                if aoi_geom is not None and aoi_area:
                    try:
                        # Calculate combined coverage
                        merged_area = merged.area if merged.area > 0 else 0
                        combined_coverage_pct = (merged_area / aoi_area) * 100 if aoi_area > 0 else 0
                        logger.info(f"📊 Combined coverage from all {len(items)} scenes: {combined_coverage_pct:.2f}%")
                        
                        # Calculate missing coverage areas
                        try:
                            missing_geom = aoi_geom.difference(merged)
                            if not missing_geom.is_empty:
                                missing_area = missing_geom.area
                                missing_coverage_pct = (missing_area / aoi_area) * 100 if aoi_area > 0 else 0
                                logger.warning(f"⚠️  Missing coverage: {missing_coverage_pct:.2f}% ({missing_area:.6f} deg²)")
                                
                                # Save missing coverage areas as GeoJSON
                                missing_path = out_dir / f"missing_coverage_{safe_time}.geojson"
                                missing_dict = mapping(missing_geom)
                                cleaned_missing = _clean_geometry_dict(missing_dict)
                                with open(missing_path, "w", encoding="utf-8") as f:
                                    json.dump({
                                        "type": "Feature",
                                        "geometry": cleaned_missing,
                                        "properties": {
                                            "missing_area_deg2": missing_area,
                                            "missing_coverage_pct": missing_coverage_pct,
                                            "aoi_total_area_deg2": aoi_area,
                                            "combined_coverage_pct": combined_coverage_pct
                                        }
                                    }, f)
                                logger.info(f"📁 Saved missing coverage areas to: {missing_path}")
                                
                                # Log bounds of missing areas for debugging
                                if hasattr(missing_geom, 'bounds'):
                                    bounds = missing_geom.bounds
                                    logger.info(f"   Missing area bounds: [{bounds[0]:.6f}, {bounds[1]:.6f}, {bounds[2]:.6f}, {bounds[3]:.6f}]")
                            else:
                                logger.info("✅ Full AOI coverage achieved with available scenes!")
                        except Exception as e:
                            logger.warning(f"Could not calculate missing coverage areas: {e}")
                    except Exception as e:
                        logger.warning(f"Could not calculate combined coverage: {e}")
            except Exception as e:
                logger.warning(f"Failed to build merged tiles geometry: {e}")
    except Exception as e:
        logger.warning(f"Failed to create tiles maps: {e}")

    # Group items by MGRS tile and select best scene per tile
    tile_to_items: Dict[str, List] = {}
    tile_cov_pct: Dict[str, float] = {}
    for it in items:
        try:
            tile_id = _extract_mgrs_tile(it.id)
        except Exception as e:
            logger.warning(f"Could not extract MGRS tile from {it.id}: {e}")
            tile_id = it.id
        tile_to_items.setdefault(tile_id, []).append(it)

        # Track max AOI coverage per tile
        cov_pct = None
        try:
            if aoi_geom is not None and aoi_area and it.geometry is not None:
                it_geom = shape(it.geometry)
                inter_area = it_geom.intersection(aoi_geom).area
                cov_pct = (inter_area / aoi_area) * 100 if aoi_area else None
        except Exception:
            cov_pct = None
        if cov_pct is not None:
            tile_cov_pct[tile_id] = max(tile_cov_pct.get(tile_id, 0.0), float(cov_pct))
    
    # If a single tile already covers the AOI (~100%), we can skip intersecting extras
    # unless we explicitly want all tiles (e.g., time-series change detection wants stable tile coverage).
    if not download_all_tiles:
        full_cover_tiles = [t for t, pct in tile_cov_pct.items() if pct >= 99.9]
        if full_cover_tiles:
            best_tile = sorted(full_cover_tiles, key=lambda t: tile_cov_pct[t], reverse=True)[0]
            logger.info(
                f"✅ AOI fully covered by tile {best_tile} ({tile_cov_pct[best_tile]:.2f}%). "
                "Skipping other intersecting tiles."
            )
        tile_to_items = {best_tile: tile_to_items[best_tile]}
    
    logger.info(f"📊 Found {len(tile_to_items)} unique MGRS tiles covering the AOI")
    
    # Parse expected date range from time parameter for validation
    expected_start = None
    expected_end = None
    try:
        if "/" in time:
            start_str, end_str = time.split("/", 1)
            expected_start = datetime.fromisoformat(start_str.replace("Z", "+00:00").replace("T", " ").split(".")[0])
            expected_end = datetime.fromisoformat(end_str.replace("Z", "+00:00").replace("T", " ").split(".")[0])
        else:
            # Single date
            expected_start = datetime.fromisoformat(time.replace("Z", "+00:00").replace("T", " ").split(".")[0])
            expected_end = expected_start
    except Exception as e:
        logger.debug(f"Could not parse date range for validation: {e}")
    
    # Select best scene per tile (lowest cloud cover)
    selected_items: List = []
    for tile_id, tile_items in tile_to_items.items():
        if pick_latest:
            best_item = sorted(tile_items, key=lambda x: x.properties.get("datetime", ""), reverse=True)[0]
        else:
            # Choose lowest cloud cover for this tile
            best_item = min(tile_items, key=lambda it: it.properties.get("eo:cloud_cover", 1e9))
        
        cc = best_item.properties.get("eo:cloud_cover", "N/A")
        dt_str = best_item.properties.get("datetime") or best_item.properties.get("start_datetime", "")
        logger.info(f"  Tile {tile_id}: Selected {best_item.id} (cloud: {cc}%, datetime: {dt_str})")
        
        # Validate scene date matches expected range
        if expected_start and expected_end and dt_str:
            try:
                scene_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00").replace("T", " ").split(".")[0])
                scene_date = scene_dt.date()
                if scene_date < expected_start.date() or scene_date > expected_end.date():
                    logger.warning(
                        f"⚠️  WARNING: Selected scene {best_item.id} has date {scene_date} "
                        f"which is OUTSIDE expected range {expected_start.date()} to {expected_end.date()}. "
                        f"This may indicate a STAC query issue or cached data from a previous run."
                    )
            except Exception as e:
                logger.debug(f"Could not validate scene date: {e}")
        
        selected_items.append((tile_id, best_item))
    
    if not selected_items:
        raise RuntimeError("No items selected for download")
    
    logger.info(f"✅ Will download {len(selected_items)} scenes (one per tile)")

    if dry_run:
        # Preflight mode: report what would be downloaded (scene per tile) but do not download anything.
        # This is useful in time-series mode to compare current vs previous scene selection before any transfer.
        tile_results: List[Dict[str, str]] = []
        for tile_id, item in selected_items:
            item_id = item.id
            base_stack_name = f"{item_id}_stack_{'-'.join(bands)}.tif"
            final_name = base_stack_name.replace(".tif", "_clipped.tif") if clip_to_aoi else base_stack_name
            stacked_path = str(Path(output_dir) / item_id / final_name)

            # Extract metadata from item
            cloud_cover = item.properties.get("eo:cloud_cover")
            datetime_str = item.properties.get("datetime") or item.properties.get("start_datetime", "")
            # Calculate AOI coverage for this item if possible
            aoi_coverage = None
            try:
                if aoi_geom is not None and aoi_area and item.geometry is not None:
                    it_geom = shape(item.geometry)
                    inter_area = it_geom.intersection(aoi_geom).area
                    aoi_coverage = (inter_area / aoi_area) * 100 if aoi_area else None
            except Exception:
                pass
            
            tile_results.append(
                {
                    "tile_id": tile_id,
                    "item_id": item_id,
                    "stacked_path": stacked_path,
                    "cloud_cover": cloud_cover,
                    "datetime": datetime_str,
                    "aoi_coverage_pct": aoi_coverage,
                }
            )

        primary_tile_id, primary_item = selected_items[0]
        primary_item_id = primary_item.id
        primary_stacked = tile_results[0]["stacked_path"]

        logger.info("[Preflight] Selected scenes (no download performed):")
        for tile_id, item in selected_items:
            cc = item.properties.get("eo:cloud_cover", "N/A")
            dt = item.properties.get("datetime") or item.properties.get("start_datetime", "")
            logger.info(f"  Tile {tile_id}: {item.id} (cloud: {cc}%, datetime: {dt})")
        
        # Generate summary report even for dry run
        _generate_imagery_summary_report(tile_results, output_dir, bands)

        return Sentinel2DownloadResult(
            item_id=primary_item_id,
            band_files={},
            stacked_path=primary_stacked,
            tiles_fc_path=None,
            tiles_merged_path=None,
            tile_results=tile_results,
        )
    
    # For backward compatibility, use first selected item as primary
    primary_tile_id, primary_item = selected_items[0]
    item_id = primary_item.id

    # Download all selected scenes (one per tile)
    tile_results: List[Dict[str, str]] = []
    all_band_files: Dict[str, str] = {}
    
    for tile_idx, (tile_id, item) in enumerate(selected_items):
        item_id = item.id
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing tile {tile_idx + 1}/{len(selected_items)}: {tile_id} (scene: {item_id})")
        logger.info(f"{'='*80}")

        item_out = out_dir / item_id
        item_out.mkdir(parents=True, exist_ok=True)

        band_to_path: Dict[str, str] = {}

        # Download bands locally (including SCL for cloud masking)
        bands_to_download = bands.copy()
        # Always download SCL for cloud masking, even if not in bands list
        if "SCL" not in bands_to_download:
            bands_to_download.append("SCL")
        
        for band in bands_to_download:
            asset = item.assets.get(band)
            resolved_key = band
            if asset is None:
                alt = BAND_ALIAS.get(band)
                if alt and item.assets.get(alt):
                    asset = item.assets[alt]
                    resolved_key = alt
            if asset is None:
                rev = REVERSE_BAND_ALIAS.get(band)
                if rev and item.assets.get(rev):
                    asset = item.assets[rev]
                    resolved_key = rev
            if asset is None:
                # SCL is optional for cloud masking, don't fail if not available
                if band == "SCL":
                    logger.warning(f"SCL band not available for cloud masking on item {item_id}. Continuing without cloud mask.")
                    continue
                available = ", ".join(sorted(item.assets.keys()))
                raise RuntimeError(f"Band asset {band} not available on item {item_id}. Available: {available}")

            href = _http_href_from_s3(asset.href)
            dst = item_out / f"{item_id}_{resolved_key}.tif"

            # Download or verify band file on disk
            if not dst.exists():
                # Fresh download with post-download integrity check
                for attempt in range(2):
                    if attempt > 0:
                        logger.info(f"Retrying download for {resolved_key} (attempt {attempt + 1})")
                    logger.info(f"Downloading {resolved_key} -> {dst}")
                    _download_file(href, dst)
                    try:
                        with rasterio.open(dst) as _ds:
                            # Read a tiny window to trigger TIFF decode without loading everything
                            _ = _ds.read(1, window=((0, 1), (0, 1)))
                        break  # success
                    except RasterioIOError as rio_err:
                        logger.warning(
                            f"Downloaded band file appears corrupted ({rio_err}). "
                            f"Will re-download {resolved_key}."
                        )
                        try:
                            dst.unlink(missing_ok=True)
                        except OSError:
                            pass
                else:
                    # Both attempts failed
                    raise RuntimeError(
                        f"Failed to download a valid TIFF for band {resolved_key} at {dst}. "
                        "See previous Rasterio errors for details."
                    )
            else:
                logger.info(f"Exists, skipping: {dst}")
                # Quick integrity check for previously downloaded files
                try:
                    with rasterio.open(dst) as _ds:
                        # Read a small window to trigger TIFF decode without loading everything
                        _ = _ds.read(1, window=((0, 1), (0, 1)))
                except RasterioIOError as rio_err:
                    logger.warning(
                        f"Existing band file appears corrupted ({rio_err}). "
                        f"Re-downloading {resolved_key} -> {dst}"
                    )
                    try:
                        dst.unlink(missing_ok=True)
                    except OSError:
                        pass
                    _download_file(href, dst)

            band_to_path[band] = str(dst)

        ordered_paths = [Path(band_to_path[band]) for band in bands]
        stacked_path = item_out / f"{item_id}_stack_{'-'.join(bands)}.tif"

        if not stacked_path.exists():
            logger.info(f"Stacking bands into: {stacked_path}")
            _stack_bands_to_multitiff(ordered_paths, stacked_path)
        else:
            logger.info(f"Stack exists locally: {stacked_path}")
        
        # Apply cloud masking if SCL band is available
        scl_path = band_to_path.get("SCL")
        if scl_path and Path(scl_path).exists():
            logger.info(f"Applying cloud mask from SCL band to stacked imagery...")
            try:
                _apply_cloud_mask_to_stack(stacked_path, scl_path)
            except Exception as cloud_err:
                logger.warning(f"Failed to apply cloud mask: {cloud_err}. Continuing without cloud masking.")
        else:
            logger.debug("SCL band not available, skipping cloud masking")

        # Clip to AOI if requested
        if clip_to_aoi:
            try:
                clipped_path = Path(str(stacked_path).replace(".tif", "_clipped.tif"))
                if GDAL_AVAILABLE:
                    logger.info("Fast clipping to AOI using GDAL Warp...")
                    cutline_geojson = {
                        "type": "FeatureCollection",
                        "features": [{"type": "Feature", "properties": {}, "geometry": geom}],
                    }
                    cutline_path = str(clipped_path.parent / f"cutline_{item_id}.geojson")
                    with open(cutline_path, "w") as fh:
                        json.dump(cutline_geojson, fh)
                    try:
                        with rasterio.open(stacked_path) as ds:
                            src_crs = ds.crs.to_string() if ds.crs else "EPSG:4326"
                        warp_opts = gdal.WarpOptions(
                            format="GTiff",
                            cutlineDSName=cutline_path,
                            cropToCutline=True,
                            dstSRS=src_crs,
                            multithread=True,
                            creationOptions=[
                                "COMPRESS=LZW",
                                "PREDICTOR=2",
                                "TILED=YES",
                                "BLOCKXSIZE=512",
                                "BLOCKYSIZE=512",
                                "BIGTIFF=YES",
                            ],
                            warpOptions=["NUM_THREADS=ALL_CPUS"],
                        )
                        out_ds = gdal.Warp(str(clipped_path), str(stacked_path), options=warp_opts)
                        if out_ds is None:
                            raise RuntimeError("GDAL Warp failed")
                        out_ds = None
                        with rasterio.open(clipped_path, "r+") as dst:
                            for band_index, band in enumerate(bands, start=1):
                                alias = BAND_ALIAS.get(band, band)
                                dst.set_band_description(band_index, f"{band} - {alias}")
                        logger.info(f"✅ Fast clipped stack written: {clipped_path}")
                    finally:
                        try:
                            Path(cutline_path).unlink(missing_ok=True)
                        except Exception:
                            pass
                else:
                    logger.info("Clipping to AOI using rasterio.mask (GDAL not available)...")
                    with rasterio.open(stacked_path) as ds:
                        src_crs = ds.crs
                        if src_crs is None:
                            raise RuntimeError("Stacked raster has no CRS; cannot clip to AOI")
                        geom_in_dst = transform_geom("EPSG:4326", src_crs.to_string(), geom)
                        data, out_transform = rio_mask(ds, [geom_in_dst], crop=True)
                        out_meta = ds.meta.copy()
                        out_meta.update(
                            {"height": data.shape[1], "width": data.shape[2], "transform": out_transform}
                        )
                    with rasterio.open(clipped_path, "w", **out_meta) as dst:
                        dst.write(data)
                        for band_index, band in enumerate(bands, start=1):
                            alias = BAND_ALIAS.get(band, band)
                            dst.set_band_description(band_index, f"{band} - {alias}")
                        logger.info(f"Clipped stack written: {clipped_path}")
                stacked_path.unlink(missing_ok=True)
                stacked_path = clipped_path
            except Exception as clip_err:
                logger.warning(f"Clipping to AOI failed; keeping full scene: {clip_err}")

        tile_results.append(
            {
                "tile_id": tile_id,
                "item_id": item_id,
                "stacked_path": str(stacked_path),
                "cloud_cover": cloud_cover,
                "datetime": datetime_str,
                "aoi_coverage_pct": aoi_coverage,
            }
        )
        all_band_files.update(band_to_path)
    
    
    # Return result with primary item (first tile) and all tile results
    primary_tile_id, primary_item = selected_items[0]
    primary_item_id = primary_item.id
    primary_stacked = tile_results[0]["stacked_path"] if tile_results else None
    
    tiles_fc_final = str(tiles_fc_path) if tiles_fc_path and Path(tiles_fc_path).exists() else None
    tiles_merged_final = str(tiles_merged_path) if tiles_merged_path and Path(tiles_merged_path).exists() else None
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Downloaded {len(tile_results)} scenes covering {len(tile_to_items)} tiles")
    for tr in tile_results:
        cc = tr.get("cloud_cover", "N/A")
        dt = tr.get("datetime", "N/A")
        cov = tr.get("aoi_coverage_pct")
        cov_str = f", AOI coverage: {cov:.2f}%" if cov is not None else ""
        logger.info(f"   Tile {tr['tile_id']}: {tr['item_id']} (cloud: {cc}%, date: {dt}{cov_str})")
    logger.info(f"{'='*80}\n")
    
    # Generate and save imagery quality summary report
    _generate_imagery_summary_report(tile_results, output_dir, bands)

    return Sentinel2DownloadResult(
        item_id=primary_item_id,
        band_files=all_band_files,
        stacked_path=primary_stacked,
        tiles_fc_path=tiles_fc_final,
        tiles_merged_path=tiles_merged_final,
        tile_results=tile_results,
    )


if __name__ == "__main__":
    # Uses the user variables defined at the top of this file
    logger.info("Starting Sentinel-2 download with user variables...")
    result = download_sentinel2(
        time=USER_TIME_RANGE,
        aoi=USER_AOI,
        output_dir=USER_OUTPUT_DIR,
        bands=USER_BANDS,
        max_cloud_cover=USER_MAX_CLOUD_COVER,
        pick_latest=USER_PICK_LATEST,
        dry_run=False,
        additional_query=USER_STAC_QUERY,
        keep_individual_bands=USER_KEEP_INDIVIDUAL_BANDS,
        clip_to_aoi=USER_CLIP_TO_AOI,
    )
    logger.info(f"Downloaded item: {result.item_id}")
    logger.info(f"Stacked file: {result.stacked_path}")
