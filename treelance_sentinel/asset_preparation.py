import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, shape, mapping
import shapely
import os
from loguru import logger
import argparse
import math
import importlib
import importlib.util
from treelance_sentinel.utils import setup_logger
import pyproj
from shapely.ops import transform
import pandas as pd

risk_analysis_spec = importlib.util.find_spec("treelance_sentinel.risk_analysis")
if risk_analysis_spec is not None:
    risk_module = importlib.import_module("treelance_sentinel.risk_analysis")
    risk_analysis_main = getattr(risk_module, "risk_analysis_main", None)
else:  # pragma: no cover - optional dependency
    risk_analysis_main = None

# Configure Loguru
logger.remove()
setup_logger()

def split_line_into_segments(line, segment_length=100):
    if line.length <= segment_length:
        return [line]
    
    segments = []
    current_length = 0
    while current_length < line.length:
        segment = shapely.ops.substring(line, current_length, current_length + segment_length)
        if segment.is_empty:
            logger.warning(f"Generated an empty segment from {current_length} to {current_length + segment_length}.")
        else:
            logger.debug(f"Generated valid segment from {current_length} to {current_length + segment_length}.")
            segments.append(segment)
        current_length += segment_length
    
    return segments

def create_buffer(gdf: gpd.GeoDataFrame, buffer_distance: float) -> gpd.GeoDataFrame:
    """
    Create a buffer around the geometries in the GeoDataFrame.
    
    Args:
        gdf: Input GeoDataFrame
        buffer_distance: Buffer distance in meters
    
    Returns:
        GeoDataFrame with buffered geometries
    """
    # Ensure the GeoDataFrame is in a projected CRS (meters)
    if gdf.crs is None or gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=3857)  # Web Mercator projection
    
    # Create buffer
    buffered = gdf.copy()
    buffered.geometry = gdf.geometry.buffer(buffer_distance)
    
    # Return to original CRS if it was geographic
    if gdf.crs.is_geographic:
        buffered = buffered.to_crs(gdf.crs)
    
    return buffered


def _resolve_tile_id(tile_row: pd.Series, fallback_index: int) -> str:
    """
    Determine a stable identifier for a tile based on available properties.
    Falls back to tile_<index> if no known property exists.
    """
    candidate_keys = (
        "id",
        "tile_id",
        "tileId",
        "name",
        "mgrs",
        "MGRS_TILE",
        "tile",
    )
    for key in candidate_keys:
        if key in tile_row:
            value = tile_row.get(key)
            if value is not None and value == value:  # not NaN
                value_str = str(value).strip()
                if value_str:
                    safe = value_str.replace("/", "_").replace(" ", "_")
                    return safe
    return f"tile_{fallback_index}"

def prepare_asset_data(input_file_path, output_dir, tile_grid_path, buffer_distance=50, segment_length=100):
    """
    Prepare asset data by creating buffers and segments.
    
    Args:
        input_file_path: Path to input line file
        output_dir: Directory to save outputs
        tile_grid_path: Path to tile grid file
        buffer_distance: Distance for buffer creation in meters
        segment_length: Length of segments in meters
    """
    if isinstance(output_dir, str) and output_dir.startswith("s3://"):
        raise ValueError("S3 output paths are no longer supported. Please use a local output directory.")

    if isinstance(input_file_path, str) and input_file_path.startswith("s3://"):
        raise ValueError("S3 AOI inputs are no longer supported. Please use a local file path.")

    if isinstance(tile_grid_path, str) and tile_grid_path.startswith("s3://"):
        raise ValueError("S3 tile grid inputs are no longer supported. Please use a local file path.")

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(
            f"AOI input file not found: {input_file_path}. Local-file-only mode no longer supports database identifiers or S3 paths."
        )

    if not os.path.exists(tile_grid_path):
        raise FileNotFoundError(f"Tile grid file not found: {tile_grid_path}")

    os.makedirs(output_dir, exist_ok=True)
    tiles_dir = os.path.join(output_dir, 'tiles')
    os.makedirs(tiles_dir, exist_ok=True)

    input_gdf = gpd.read_file(input_file_path)
    
    # Read tile grid
    tile_grid = gpd.read_file(tile_grid_path)
    
    # Ensure both are in the same CRS (WGS84)
    if input_gdf.crs != tile_grid.crs:
        logger.info(f"Reprojecting input from {input_gdf.crs} to {tile_grid.crs}")
        input_gdf = input_gdf.to_crs(tile_grid.crs)
    
    # Create buffer in Web Mercator (for accurate distance)
    buffer_gdf = input_gdf.to_crs(epsg=3857)
    buffer_gdf = create_buffer(buffer_gdf, buffer_distance)
    buffer_gdf = buffer_gdf.to_crs(tile_grid.crs)  # Back to WGS84
    
    # Find tiles that intersect with the buffer
    intersecting_tiles = []
    for idx, buffer_geom in buffer_gdf.iterrows():
        # Find tiles that intersect with this buffer geometry
        tiles = tile_grid[tile_grid.intersects(buffer_geom.geometry)]
        intersecting_tiles.append(tiles)
    
    # Combine all intersecting tiles and remove duplicates
    if intersecting_tiles:
        intersecting_tiles = pd.concat(intersecting_tiles)
        intersecting_tiles = intersecting_tiles[~intersecting_tiles.index.duplicated(keep='first')]
    else:
        intersecting_tiles = tile_grid.iloc[0:0]  # Empty GeoDataFrame with same structure
    
    logger.info(f"Found {len(intersecting_tiles)} intersecting tiles")
    for idx, tile in intersecting_tiles.iterrows():
        tile_id = _resolve_tile_id(tile, idx)
        logger.info(f"Tile {tile_id}")
    
    # Save segmented lines
    segments_output = os.path.join(output_dir, 'segments.geojson')
    segmented_gdf = gpd.GeoDataFrame(geometry=split_line_into_segments(input_gdf.geometry.iloc[0]), crs=input_gdf.crs)
    segmented_gdf.to_file(segments_output, driver='GeoJSON')
    logger.info(f"Saved segmented lines to: {segments_output}")
    
    # Save buffer
    buffer_output = os.path.join(output_dir, 'buffer.geojson')
    buffer_gdf.to_file(buffer_output, driver='GeoJSON')
    logger.info(f"Saved buffer to: {buffer_output}")
    
    # Save intersecting tiles
    tiles_output = os.path.join(output_dir, 'intersecting_tiles.geojson')
    intersecting_tiles.to_file(tiles_output, driver='GeoJSON')
    logger.info(f"Saved intersecting tiles to: {tiles_output}")
    
    # Create tiled buffers
    tiled_buffers = {}
    for idx, tile in intersecting_tiles.iterrows():
        tile_id = _resolve_tile_id(tile, idx)
        
        # Clip buffer by tile geometry
        tile_geom = tile.geometry
        tile_buffer = gpd.clip(buffer_gdf, tile_geom)
        
        # Save individual tile buffer in tiles subdirectory
        tile_buffer_output = os.path.join(tiles_dir, f'buffer_{tile_id}.geojson')
        tile_buffer.to_file(tile_buffer_output, driver='GeoJSON')
        tiled_buffers[tile_id] = tile_buffer_output
        logger.info(f"Saved tiled buffer for {tile_id} to: {tile_buffer_output}")

    # After creating tiled buffers
    tiled_assets = {}
    for idx, tile in intersecting_tiles.iterrows():
        tile_id = _resolve_tile_id(tile, idx)
        
        # Clip original geometries by tile geometry
        tile_asset = gpd.clip(input_gdf, tile.geometry)
        
        # Split the clipped geometries into segments
        segmented_tile_geometries = []
        for geom in tile_asset.geometry:
            segments = split_line_into_segments(geom, segment_length)
            segmented_tile_geometries.extend(segments)
        
        # Create GeoDataFrame for segmented tile
        segmented_tile_gdf = gpd.GeoDataFrame(geometry=segmented_tile_geometries, crs=input_gdf.crs)
        
        # Save segmented tile
        tile_asset_output = os.path.join(tiles_dir, f'segments_{tile_id}.geojson')
        segmented_tile_gdf.to_file(tile_asset_output, driver='GeoJSON')
        tiled_assets[tile_id] = tile_asset_output
        logger.info(f"Saved segmented tile for {tile_id} to: {tile_asset_output}")
    
    return {
        'buffer_path': buffer_output,
        'segments_path': segments_output,
        'intersecting_tiles_path': tiles_output,
        'tiled_buffers': tiled_buffers,
        'tiled_assets': tiled_assets
    }

def run_risk_analysis_on_tiled_assets(tiled_assets, predictions_dir, output_dir, thresholds):
    if risk_analysis_main is None:
        logger.warning("risk_analysis_main is not available; skipping risk analysis step.")
        return

    for tile_id, asset_path in tiled_assets.items():
        predicted_polygon_path = os.path.join(predictions_dir, f"{tile_id}_predictions.gpkg")
        if not os.path.exists(predicted_polygon_path):
            logger.warning(f"No predicted polygon file found for tile {tile_id}")
            continue

        output_path = os.path.join(output_dir, f"{tile_id}_risk_analysis.gpkg")
        risk_analysis_main(
            line_gpkg_path=asset_path,
            polygon_gpkg_path=predicted_polygon_path,
            output_path=output_path,
            thresholds=thresholds
        )
        logger.info(f"Completed risk analysis for tile {tile_id}, output saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process line data for asset preparation')
    parser.add_argument('--input', required=True, help='Input line file path')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--tile-grid', required=True, help='Path to Sentinel-2 tile grid file')
    parser.add_argument('--buffer-distance', type=float, default=50, help='Buffer distance in meters')
    parser.add_argument('--segment-length', type=float, default=100, help='Segment length in meters')
    
    args = parser.parse_args()
    
    try:
        result = prepare_asset_data(
            args.input,
            args.output_dir,
            args.tile_grid,
            args.buffer_distance,
            args.segment_length
        )
        if result:
            logger.info("Asset data preparation completed successfully!")
            run_risk_analysis_on_tiled_assets(
                tiled_assets=result['tiled_assets'],
                predictions_dir='/path/to/predictions',
                output_dir='/path/to/risk_analysis_output',
                thresholds={'high': 1, 'medium': 3, 'low': 5}
            )
        else:
            logger.error("Asset data preparation failed!")
    except Exception as e:
        logger.error(f"Error during asset data preparation: {e}")
        raise