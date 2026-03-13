import os
import glob
import math
import shutil
import numpy as np
import warnings
import logging
from osgeo import gdal
from osgeo import osr
import multiprocessing
from functools import partial
import pyproj

# Configure logging to filter out GDAL warnings
logging.getLogger('rasterio._base').setLevel(logging.ERROR)
logging.getLogger('rasterio._env').setLevel(logging.ERROR)
logging.getLogger('rasterio._io').setLevel(logging.ERROR)
logging.getLogger('rasterio._warp').setLevel(logging.ERROR)
logging.getLogger('rasterio._crs').setLevel(logging.ERROR)
logging.getLogger('rasterio._transform').setLevel(logging.ERROR)
logging.getLogger('rasterio._features').setLevel(logging.ERROR)
logging.getLogger('rasterio._fill').setLevel(logging.ERROR)

# Suppress all GDAL warnings
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.SetConfigOption('CPL_LOG', 'NONE')
gdal.SetConfigOption('CPL_DEBUG', 'OFF')
gdal.SetConfigOption('GTIFF_SRS_SOURCE', 'EPSG')

# Suppress Python warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set environment variables
os.environ['PROJ_LIB'] = '/usr/share/proj'
os.environ['GDAL_DATA'] = '/usr/share/gdal'
os.environ['GDAL_SKIP'] = ''
os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif,.tiff,.geojson,.json,.gpkg')
os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')

from loguru import logger
from treelance_sentinel.utils import timer_decorator, Timer, setup_logger
import rasterio
from rasterio.mask import mask
from typing import List, Dict, Optional, Union, Set
from shapely.geometry import mapping, box, Polygon
import geopandas as gpd
from shapely.geometry import shape
import zipfile
from rasterio.crs import CRS


def _extract_tile_id_from_filename(name: str) -> Optional[str]:
    """Best-effort extraction of the Sentinel tile token from a raster name."""
    parts = name.split("_")
    return parts[1] if len(parts) >= 2 else None


def _build_buffer_candidates(base_name: str, tile_id: Optional[str]) -> List[str]:
    """Return ordered candidate identifiers that may match a buffer filename."""
    candidates: List[str] = []
    if tile_id:
        candidates.append(tile_id)
    candidates.append(base_name)
    if "_stack" in base_name:
        candidates.append(base_name.split("_stack")[0])
    if base_name.endswith("_clipped"):
        candidates.append(base_name[:-len("_clipped")])
    return list(dict.fromkeys(candidates))


def _collect_available_buffer_ids(buffer_base_dir: str) -> Set[str]:
    """List all buffer identifiers present under the given local directory."""
    buffer_ids: Set[str] = set()
    pattern = os.path.join(buffer_base_dir, "buffer_*.geojson")
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        if name.startswith("buffer_") and name.endswith(".geojson"):
            buffer_ids.add(name[len("buffer_"):-len(".geojson")])
    return buffer_ids


@timer_decorator
def extract_zip_files(input_dir: str, extract_dir: str):
    """Extract all zip files from the input directory."""
    for filename in os.listdir(input_dir):
        if filename.endswith('.zip'):
            zip_path = os.path.join(input_dir, filename)
            logger.info(f"Checking zip file: {zip_path}")

            # Verify file exists and has size
            if not os.path.exists(zip_path):
                logger.error(f"Zip file does not exist: {zip_path}")
                continue

            file_size = os.path.getsize(zip_path)
            if file_size == 0:
                logger.error(f"Zip file is empty: {zip_path}")
                continue

            try:
                with Timer(f"Extracting {filename}"):
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
            except zipfile.BadZipFile:
                logger.error(f"Invalid or corrupted zip file: {zip_path}")
            except Exception as e:
                logger.error(f"Error extracting {zip_path}: {str(e)}")


@timer_decorator
def process_single_scene(scene_dir: str, scene_tile_id: str, tile_aoi: gpd.GeoDataFrame, 
                        temp_extract_folder: str, merged_output_dir: str, 
                        final_clipped_output_dir: str, bands_of_interest: list,
                        tiled_buffer_path: str = None):
    """Process a single scene and calculate EVI and NDVI."""
    logger.info(f"Processing scene directory: {scene_dir}")

    # Use tiled buffer if provided, otherwise use tile_aoi
    if tiled_buffer_path and os.path.exists(tiled_buffer_path):
        logger.info(f"Using tiled buffer from: {tiled_buffer_path}")
        clip_geometry = gpd.read_file(tiled_buffer_path)
        temp_aoi_path = tiled_buffer_path
    else:
        logger.info("Using tile AOI for clipping")
        clip_geometry = tile_aoi
        temp_aoi_path = os.path.join(temp_extract_folder, f"{scene_tile_id}_aoi.geojson")
        clip_geometry.to_file(temp_aoi_path)

    granule_dir = os.path.join(scene_dir, 'GRANULE')
    if os.path.exists(granule_dir):
        granule_contents = os.listdir(granule_dir)
        if granule_contents:
            tile_dir = os.path.join(granule_dir, granule_contents[0])
            img_data_dir = os.path.join(tile_dir, 'IMG_DATA')

            # Temporary list to store band paths for merging
            band_paths = []

            for band in bands_of_interest:
                search_pattern = os.path.join(img_data_dir, 'R10m', f'*{band}')
                band_path = glob.glob(search_pattern)

                if band_path:
                    logger.info(f"Found band file: {band_path[0]}")
                    band_paths.append(band_path[0])
                else:
                    logger.warning(f"Band {band} not found using pattern: {search_pattern}")
                    return None  # Alternatively, decide how to handle missing bands

            # Create final output path
            final_output = os.path.join(final_clipped_output_dir, f"{scene_tile_id}.tif")

            try:
                # Merge all bands into a single raster in the desired order: B04, B03, B02, B08
                vrt_path = os.path.join(temp_extract_folder, f"{scene_tile_id}_merged.vrt")
                vrt_options = gdal.BuildVRTOptions(separate=True)
                gdal.BuildVRT(vrt_path, band_paths, options=vrt_options)

                # Clip the merged raster using the AOI
                logger.info(f"Clipping merged raster with buffer from: {temp_aoi_path}")
                clipped_path = os.path.join(temp_extract_folder, f"{scene_tile_id}_clipped.tif")
                gdal.Warp(
                    clipped_path,
                    vrt_path,
                    cutlineDSName=temp_aoi_path,
                    cropToCutline=True,
                    dstNodata=65535,
                    creationOptions=["COMPRESS=DEFLATE"]
                )

                # Clean up temporary VRT
                os.remove(vrt_path)

                logger.info(f"Successfully clipped raster: {clipped_path}")

                # Calculate EVI and NDVI and append to the clipped raster
                calculate_indices(clipped_path, final_output, debug_mode=False)

                # Clean up temporary files
                os.remove(clipped_path)

                logger.info(f"Successfully created output with indices: {final_output}")
                return final_output

            except Exception as e:
                logger.error(f"Error creating raster with indices: {str(e)}")
                return None


def calculate_indices(input_raster: str, output_raster: str, debug_mode: bool = False):
    """
    Legacy hook for index augmentation.

    Historically this function computed and appended Planet-style indices
    (NDVI, EVI, NDBI, NDRE) to the input raster. The current Sentinel-based
    Treelance workflow no longer relies on those derived bands at this stage:

    - NDVI / NDMI needed for vitality are computed later in the time-series
      and change-detection pipeline (see `_calculate_ndmi_and_add_ndvi`
      in `treelance_sentinel.cli`).
    - EVI / NDBI / NDRE are not part of the production Sentinel pipeline.

    To keep the public API stable while removing unused complexity, this
    function now performs a simple pass-through copy:

        input_raster  →  output_raster (same bands, metadata, no extra indices)
    """
    logger.info(
        "Index augmentation is disabled in imagery_preprocessing; "
        "copying input raster to output without adding NDVI/EVI/NDBI/NDRE.\n"
        f"  input={input_raster}\n"
        f"  output={output_raster}"
    )

    try:
        import os
        import shutil

        abs_in = os.path.abspath(input_raster)
        abs_out = os.path.abspath(output_raster)

        if abs_in == abs_out:
            logger.warning(
                "calculate_indices called with identical input and output paths; "
                "skipping copy."
            )
            return

        os.makedirs(os.path.dirname(abs_out), exist_ok=True)
        shutil.copy2(abs_in, abs_out)
        logger.info("Copied raster without index augmentation.")
    except Exception as exc:
        logger.error(f"Failed to copy raster in calculate_indices: {exc}", exc_info=True)
        raise


@timer_decorator
def tile_aoi_from_imagery(input_dir: str, aoi: Dict) -> List[Dict]:
    """
    Create tiles from AOI based on downloaded imagery extents.
    
    Args:
        input_dir (str): Directory containing downloaded Planet tiles
        aoi (Dict): Area of Interest as GeoJSON
        
    Returns:
        List[Dict]: List of tile geometries as GeoJSON features
    """
    logger.info("Creating tiles from AOI based on imagery extents")
    
    # Find all Planet tiles in the input directory
    planet_tiles = glob.glob(os.path.join(input_dir, "*.tif"))
    if not planet_tiles:
        raise ValueError(f"No Planet tiles found in {input_dir}")
    
    # Create tiles based on imagery extents
    tiles = []
    for tile_path in planet_tiles:
        try:
            with rasterio.open(tile_path) as src:
                # Get the tile's bounds
                bounds = src.bounds
                
                # Create a tile geometry
                tile_geom = {
                    "type": "Polygon",
                    "coordinates": [[
                        [bounds.left, bounds.bottom],
                        [bounds.right, bounds.bottom],
                        [bounds.right, bounds.top],
                        [bounds.left, bounds.top],
                        [bounds.left, bounds.bottom]
                    ]]
                }
                
                # Create a GeoDataFrame for the tile
                tile_gdf = gpd.GeoDataFrame(
                    geometry=[shape(tile_geom)],
                    crs=src.crs
                )
                
                # Create a GeoDataFrame for the AOI
                aoi_gdf = gpd.GeoDataFrame(
                    geometry=[shape(aoi["features"][0]["geometry"])],
                    crs=src.crs
                )
                
                # Check if tile intersects with AOI
                if tile_gdf.intersects(aoi_gdf).any():
                    # Create intersection with AOI
                    intersection = tile_gdf.intersection(aoi_gdf)
                    
                    # Add to tiles list
                    tiles.append({
                        "type": "Feature",
                        "properties": {
                            "name": os.path.basename(tile_path),
                            "path": tile_path
                        },
                        "geometry": mapping(intersection.geometry[0])
                    })
                    
        except Exception as e:
            logger.error(f"Error processing tile {tile_path}: {str(e)}")
            continue
    
    logger.info(f"Created {len(tiles)} tiles from imagery extents")
    return tiles


def process_single_file(args):
    """Process a single file with all necessary parameters."""
    file_path, output_dir, aoi_gdf, buffer_base_dir, debug_mode = args
    
    filename = os.path.basename(file_path)
    logger.info(f"Processing imagery file: {filename}")
    
    # Determine processing CRS: prefer raster CRS, fallback to EPSG:3857
    target_epsg = None
    try:
        with rasterio.open(file_path) as probe:
            target_epsg = probe.crs.to_string() if probe.crs is not None else None
    except Exception:
        target_epsg = None
    if not target_epsg:
        target_epsg = "EPSG:3857"
    logger.info(f"Processing CRS: {target_epsg}")

    try:
        # Open the TIFF file
        with rasterio.open(file_path) as src:
            # Get original metadata but override CRS later
            original_meta = src.meta.copy()
            # Log the CRS found in the file, but we will ignore it.
            logger.info(f"CRS found in TIFF (will be overridden): {src.crs}")
            
            # Store the original colorinterp values for preservation
            colorinterp = src.colorinterp
            logger.info(f"Original color interpretation: {colorinterp}")
            
            # Get the bounds of the imagery
            img_bounds = box(*src.bounds)
            logger.info(f"Image bounds ({target_epsg}): {src.bounds}")
            
            output_path = os.path.join(output_dir, filename)
            output_path_local = output_path
            
            # Initialize clip_geometry with AOI as default
            # Ensure AOI GDF is in EPSG:4326 before transforming to processing CRS
            if aoi_gdf.crs != "EPSG:4326":
                 logger.warning(f"AOI CRS is not EPSG:4326 ({aoi_gdf.crs}). Assuming it's EPSG:4326.")
                 aoi_gdf = aoi_gdf.set_crs("EPSG:4326", allow_override=True)

            # Reproject AOI to the processing CRS
            aoi_reprojected = aoi_gdf.to_crs(target_epsg)
            clip_geometry = aoi_reprojected.geometry.unary_union
            source_crs_for_clip = target_epsg # Use the processing CRS directly

            # Try to use buffer file; falling back to AOI is not allowed
            if not buffer_base_dir:
                raise RuntimeError(
                    "tiled_buffer_path is required for Step 3 preprocessing; no buffer directory was provided."
                )

            if buffer_base_dir:
                base_name = os.path.splitext(filename)[0]
                tile_id = _extract_tile_id_from_filename(base_name)
                buffer_candidates = _build_buffer_candidates(base_name, tile_id)

                def _candidate_path(candidate: str) -> str:
                    return os.path.join(buffer_base_dir, f"buffer_{candidate}.geojson")

                def _try_read_buffer(path: str):
                    try:
                        return gpd.read_file(path)
                    except Exception:
                        return None

                candidate_paths = [_candidate_path(c) for c in buffer_candidates]
                logger.info(f"Looking for matching buffer file among: {candidate_paths}")

                buffer_gdf = None
                for candidate_path in candidate_paths:
                    buffer_gdf = _try_read_buffer(candidate_path)
                    if buffer_gdf is not None:
                        logger.info(f"Using local buffer tile: {candidate_path}")
                        break

                if buffer_gdf is None:
                    raise RuntimeError(
                        f"No matching buffer tile found for {filename}. Checked: {candidate_paths}. "
                        "Buffer tiles are mandatory; stopping."
                    )

                logger.info(f"Buffer CRS: {buffer_gdf.crs}")
                if buffer_gdf.crs != target_epsg:
                    logger.info(f"Reprojecting buffer from {buffer_gdf.crs} to {target_epsg}")
                    buffer_gdf = buffer_gdf.to_crs(target_epsg)
                clip_geometry = buffer_gdf.geometry.unary_union
                source_crs_for_clip = target_epsg
            
            # Prepare the final clip geometry (already in processing CRS)
            # No further transformation needed as both AOI and buffer (if used) are in processing CRS
            logger.info(f"Using clip geometry in {source_crs_for_clip}")
            
            # Clip imagery
            try:
                # Ensure geometry is valid
                if not clip_geometry.is_valid:
                    logger.info("Fixing invalid geometry")
                    clip_geometry = clip_geometry.buffer(0)
                
                # Convert geometry to GeoJSON format
                clip_geojson = mapping(clip_geometry)
                
                # Clip the raster using the geometry in the processing CRS
                # Rasterio mask handles the CRS matching internally if the geometry CRS is set
                # We assume the source raster 'src' is already in the processing CRS
                out_image, out_transform = mask(src, [clip_geojson], crop=True)
                
                # Update metadata: Use original metadata and set processing CRS
                out_meta = original_meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "crs": target_epsg,
                    "compress": "deflate",
                    "predictor": 2,
                    "zlevel": 9,
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                    "interleave": "band"
                })
                
                # Save the clipped raster
                with rasterio.open(output_path_local, "w", **out_meta) as dest:
                    dest.write(out_image)
                    # Preserve original color interpretation
                    dest.colorinterp = colorinterp
                
                # Calculate NDVI and EVI and replace Band 9 with NDVI and add EVI as Band 10
                logger.info(f"Calculating vegetation indices for {filename}")
                temp_output = output_path_local + "_temp.tif"
                os.rename(output_path_local, temp_output)
                try:
                    calculate_indices(temp_output, output_path_local, debug_mode=debug_mode)
                    # Remove temporary file after index calculation
                    os.remove(temp_output)
                except Exception as e:
                    logger.error(f"Error calculating indices: {str(e)}. Reverting to original file.")
                    os.rename(temp_output, output_path_local)

                # Context stacking disabled (stack_context_to_ps removed)
                
                # Verify output CRS
                with rasterio.open(output_path_local) as check:
                    # Use pyproj to compare CRS objects reliably
                    check_pyproj = pyproj.CRS.from_user_input(check.crs)
                    target_pyproj = pyproj.CRS.from_user_input(target_epsg)
                    
                    if check_pyproj.equals(target_pyproj):
                        logger.info(f"Successfully clipped {filename} with consistent CRS ({target_epsg})")
                    else:
                        logger.error(f"CRS mismatch! Expected: {target_epsg}, Output: {check.crs}")
                        logger.warning(f"Output CRS ({check.crs.to_string()}) does not match expected {target_epsg}.")
                
                return True
                
            except Exception as e:
                # Log rasterio-specific errors if available
                if "CRSError" in str(e) or "CRS" in str(e) or "WKT" in str(e):
                     logger.error(f"CRS-related error during clipping: {str(e)}")
                logger.error(f"Error clipping {filename}: {str(e)}")
                # Fallback to processing full raster without clipping
                try:
                    out_meta = original_meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": src.height,
                        "width": src.width,
                        "transform": src.transform,
                        "crs": target_epsg,
                        "compress": "deflate",
                        "predictor": 2,
                        "zlevel": 9,
                        "tiled": True,
                        "blockxsize": 256,
                        "blockysize": 256,
                        "interleave": "band"
                    })
                    with rasterio.open(output_path_local, "w", **out_meta) as dest:
                        dest.write(src.read())
                        dest.colorinterp = colorinterp
                    logger.info(f"Proceeding with full scene for {filename}")
                    temp_output = output_path_local + "_temp.tif"
                    os.rename(output_path_local, temp_output)
                    try:
                        calculate_indices(temp_output, output_path_local, debug_mode=debug_mode)
                        os.remove(temp_output)
                    except Exception as ie:
                        logger.error(f"Error calculating indices on full scene: {ie}. Reverting to original file.")
                        os.rename(temp_output, output_path_local)
                    return True
                except Exception as fe:
                    logger.error(f"Fallback processing failed: {fe}")
                    return False
                
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return False


@timer_decorator
def process_imagery(
    input_dir: str,
    output_dir: str,
    aoi: Union[str, gpd.GeoDataFrame, Dict],
    tile_size: int = 512,
    overlap: int = 0,
    tiles_path: Optional[str] = None,
    tiled_buffer_path: Optional[Union[str, Dict]] = None,
    debug_mode: bool = False,
    stacked_file: Optional[str] = None,
) -> None:
    """Process imagery by matching Planet files with their corresponding buffer files."""
    if isinstance(input_dir, str) and input_dir.startswith('s3://'):
        raise ValueError("S3 imagery inputs are no longer supported. Please use a local input directory.")
    if isinstance(output_dir, str) and output_dir.startswith("s3://"):
        raise ValueError("S3 imagery outputs are no longer supported. Please use a local output directory.")

    os.makedirs(output_dir, exist_ok=True)
    
    # Read AOI
    if isinstance(aoi, str):
        if aoi.startswith('s3://'):
            raise ValueError("S3 AOI inputs are no longer supported. Please use a local AOI file.")
        aoi_gdf = gpd.read_file(aoi)
    elif isinstance(aoi, gpd.GeoDataFrame):
        aoi_gdf = aoi.copy()
    elif isinstance(aoi, dict):
        aoi_gdf = gpd.GeoDataFrame.from_features([{
            "type": "Feature",
            "properties": {},
            "geometry": aoi
        }])
        aoi_gdf = aoi_gdf.set_crs("EPSG:4326")
    else:
        raise ValueError("AOI must be a file path, GeoDataFrame, or dictionary")
    
    # Get the base directory for buffer files
    buffer_base_dir = None
    if isinstance(tiled_buffer_path, str):
        buffer_base_dir = os.path.dirname(tiled_buffer_path)
    elif isinstance(tiled_buffer_path, dict) and tiled_buffer_path:
        first_path = next(iter(tiled_buffer_path.values()))
        buffer_base_dir = os.path.dirname(first_path)
    
    if buffer_base_dir is None:
        raise RuntimeError("tiled_buffer_path is required for Step 3 preprocessing.")
    
    available_buffer_ids = _collect_available_buffer_ids(buffer_base_dir)
    if not available_buffer_ids:
        logger.warning(f"No buffer geojson files discovered under {buffer_base_dir}.")
    
    # Get list of files to process
    files_to_process: List[str] = []
    files_to_process = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.endswith('.tif')
    ]
    
    # If no files discovered, try explicit stacked file path
    if (not files_to_process) and stacked_file:
        if isinstance(stacked_file, str) and stacked_file.startswith('s3://'):
            raise ValueError("S3 stacked_file inputs are no longer supported. Please use a local raster path.")
        files_to_process = [stacked_file]

    # Filter out rasters without a corresponding buffer tile
    filtered_files: List[str] = []
    for file_path in files_to_process:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        tile_id = _extract_tile_id_from_filename(base_name)
        candidates = _build_buffer_candidates(base_name, tile_id)
        if any(candidate in available_buffer_ids for candidate in candidates):
            filtered_files.append(file_path)
        else:
            logger.info(f"Skipping {os.path.basename(file_path)} (no matching buffer tile).")

    files_to_process = filtered_files

    if not files_to_process:
        logger.warning("No imagery files matched available buffer tiles. Nothing to process.")
        return

    # Prepare arguments for multiprocessing
    process_args = [
        (file_path, output_dir, aoi_gdf, buffer_base_dir, debug_mode)
        for file_path in files_to_process
    ]
    
    # Determine number of workers (balance CPU and I/O pressure)
    cpu_target = max(1, math.ceil(multiprocessing.cpu_count() * 0.6))
    num_workers = max(1, min(len(process_args), cpu_target))
    logger.info(f"Using {num_workers} workers for parallel processing (balanced for I/O)")
    
    # Process files in parallel using imap_unordered to avoid head-of-line blocking
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(pool.imap_unordered(process_single_file, process_args))
    
    # Log summary of results
    successful = sum(1 for r in results if r)
    failed = len(results) - successful
    logger.info(f"Processing complete. Successfully processed: {successful}, Failed: {failed}")


# Example Usage
if __name__ == "__main__":
    final_output = process_imagery(
        input_dir='/path/to/input_zips',
        output_dir='/path/to/output_temp',
        aoi={
            "type": "Polygon",
            "coordinates": [[
                [0, 0],
                [10000, 0],
                [10000, 10000],
                [0, 10000],
                [0, 0]
            ]]
        },
        tile_size=512,
        overlap=0
    )
    
    print(f"Processing complete. Final output directory: {final_output}")


class ImageryPreprocessor:
    """Class-based wrapper for imagery preprocessing functionality."""
    
    def __init__(self, input_dir: str, output_dir: str, aoi, config: dict | None = None, debug_mode: bool = False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.aoi = aoi
        self.config = config or {}
        self.debug_mode = debug_mode
        self.tile_size = self.config.get('tile_size', 512)
        self.overlap = self.config.get('overlap', 0)
    
    def process(self, tiles_path: str | None = None, tiled_buffer_path: str | dict | None = None, stacked_file: str | None = None) -> None:
        """Process imagery with the given parameters."""
        return process_imagery(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            aoi=self.aoi,
            tile_size=self.tile_size,
            overlap=self.overlap,
            tiles_path=tiles_path,
            tiled_buffer_path=tiled_buffer_path,
            debug_mode=self.debug_mode,
            stacked_file=stacked_file
        )
