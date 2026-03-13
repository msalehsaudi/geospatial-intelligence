import os
from dataclasses import dataclass
import copy
from datetime import datetime
from pathlib import Path
from loguru import logger
from typing import Any, Dict
import yaml
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.crs import CRS
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile

from treelance_sentinel.imagery_preprocessing import ImageryPreprocessor
from treelance_sentinel.imagery_processing import RasterProcessor
from treelance_sentinel.prediction import main as predict, generate_class_summary
from treelance_sentinel.visualization import (
    create_classification_map,
    create_classification_map_from_dissolved_gpkg,
    create_classification_map_from_polygons,
    create_classification_map_from_tiff,
)
from treelance_sentinel.utils import DurationCollector, Timer, setup_logger, timer_decorator
from treelance_sentinel.asset_preparation import prepare_asset_data
from treelance_sentinel.sentinel_download.sentinel_aws_downloader import (
    download_sentinel2 as s2_aws_download,
)

@dataclass(frozen=True)
class PipelineDirectories:
    asset_preparation: str
    raw_data: str
    processed_data: str
    segmentation: str
    predictions: str


class E2EClassificationPipeline:
    def __init__(self, config_path: str | None = None, config: Dict | None = None):
        """Initialize the pipeline with configuration."""
        if config is None and not config_path:
            raise ValueError("Either config_path or config must be provided")

        self.config_path = config_path or "<in-memory>"
        setup_logger()  # Configure logger with filters

        if config is not None:
            self.config = copy.deepcopy(config)
            logger.info("Configuration loaded from in-memory overrides")
        else:
            self.config = self._load_config(self.config_path)

        self._setup_logging()
        self.paths = self._setup_directories()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Extract filename without extension for automatic directory setup
            config_filename = Path(config_path).stem
            
            # Ensure directories section exists
            if "directories" not in config:
                config["directories"] = {}
            
            # Only auto-set base_output_dir if it's missing.
            # IMPORTANT: time_series mode writes temporary YAML files; using the temp filename would
            # incorrectly redirect outputs to random output prefixes.
            if "base_output_dir" not in config["directories"] or not config["directories"]["base_output_dir"]:
                base_output_dir = str(Path(config_path).resolve().parent / config_filename)
                config["directories"]["base_output_dir"] = base_output_dir
                logger.info(f"Auto-set base_output_dir to: {base_output_dir} (based on config filename)")
            else:
                logger.info(f"Using configured base_output_dir: {config['directories']['base_output_dir']}")
            
            # Ensure sentinel_aws section exists
            if "sentinel_aws" not in config:
                config["sentinel_aws"] = {}
            
            # Only auto-set sentinel_aws.output_dir if missing (same rationale as base_output_dir).
            base_output_dir = config["directories"]["base_output_dir"]
            if not config["sentinel_aws"].get("output_dir"):
                sentinel_output_dir = str(Path(base_output_dir) / "imagery")
                config["sentinel_aws"]["output_dir"] = sentinel_output_dir
                logger.info(f"Auto-set sentinel_aws.output_dir to: {sentinel_output_dir}")
            else:
                logger.info(f"Using configured sentinel_aws.output_dir: {config['sentinel_aws']['output_dir']}")
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _setup_logging(self):
        """Configure logging (console only, no log files)."""
        # Set debug mode based on config
        debug_mode = self.config.get("logging", {}).get("debug_mode", False)
        if debug_mode:
            logger.info("Debug logging enabled - detailed logs will be shown")
        else:
            logger.info("Clean logging mode - only essential logs will be shown")
        
        # Store debug mode for use in other modules
        self.debug_mode = debug_mode

    def _setup_directories(self) -> PipelineDirectories:
        """Create all necessary output directories."""
        base_dir = self.config["directories"]["base_output_dir"]
        if isinstance(base_dir, str) and base_dir.startswith("s3://"):
            raise ValueError("S3 paths are no longer supported. Please use a local base_output_dir.")

        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            logger.info(f"Created base directory: {base_dir}")

        def _resolve_path(subdir_value: str, label: str) -> str:
            if isinstance(subdir_value, str) and subdir_value.startswith("s3://"):
                raise ValueError(f"S3 paths are no longer supported for {label}. Please use a local path.")

            if os.path.isabs(subdir_value):
                path = subdir_value
            else:
                path = os.path.join(base_dir, subdir_value)

            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                logger.info(f"Created new directory for {label}: {path}")
            return path

        directories = {
            "asset_preparation": _resolve_path(self.config["directories"]["asset_preparation"], "asset_preparation"),
            "raw_data": _resolve_path(self.config["directories"]["raw_data"], "raw_data"),
            "processed_data": _resolve_path(self.config["directories"]["processed_data"], "processed_data"),
            "segmentation": _resolve_path(self.config["directories"]["segmentation"], "segmentation"),
            "predictions": _resolve_path(self.config["directories"]["predictions"], "predictions"),
        }

        return PipelineDirectories(**directories)

    def _download_sentinel_data(self) -> str:
        """Download Sentinel-2 data and return the tiles path."""
        logger.info("Step 1: Downloading Sentinel-2 data and creating tiles")
        s2_cfg = self.config.get("sentinel_aws", {})
        if not s2_cfg:
            raise ValueError("Missing 'sentinel_aws' configuration in YAML")

        try:
            aoi = (
                s2_cfg.get("aoi")
                or self.config.get("input", {}).get("aoi")
                or self.config.get("input", {}).get("grid_file")
            )
            if not aoi:
                raise ValueError("AOI not found in 'sentinel_aws.aoi' or 'input.aoi' configuration")
            # Get buffer_distance from config (default to 0 if not specified)
            buffer_distance = self.config.get("input", {}).get("buffer_distance", 0)
            
            s2_result = s2_aws_download(
                time=s2_cfg.get("time_range"),
                aoi=aoi,
                output_dir=s2_cfg.get("output_dir") or self.paths.raw_data,
                bands=s2_cfg.get("bands"),
                max_cloud_cover=int(s2_cfg.get("max_cloud_cover", 20)),
                pick_latest=not bool(s2_cfg.get("pick_lowest_cloud", True)),
                download_all_tiles=bool(s2_cfg.get("download_all_tiles", False)),
                dry_run=False,
                additional_query=s2_cfg.get("additional_query"),
                keep_individual_bands=bool(s2_cfg.get("keep_individual_bands", False)),
                clip_to_aoi=bool(s2_cfg.get("clip_to_aoi", False)),
                buffer_distance=buffer_distance,
            )
            logger.info(f"Sentinel-2 stacked file: {s2_result.stacked_path}")
            
            # Log if clipped file is being used
            if "_clipped" in s2_result.stacked_path:
                logger.info(f"✅ Using clipped imagery file for subsequent processing steps")
            
            tiles_path = s2_result.tiles_fc_path or ""
            if not tiles_path:
                raise ValueError("Tiles FeatureCollection path missing from Sentinel download result")
            return tiles_path
        except Exception as exc:
            logger.error(f"Sentinel-2 download failed: {exc}")
            raise

    def _prepare_assets(self, tiles_path: str) -> Dict[str, Any]:
        """Run asset preparation and return the resulting paths."""
        with Timer("Asset Preparation", module="asset_prep"):
            input_file = self.config["input"].get("aoi") or self.config["input"].get("grid_file")
            if not input_file:
                raise ValueError("AOI not found in 'input.aoi' or 'input.grid_file' configuration")
            asset_results = prepare_asset_data(
                input_file_path=input_file,
                output_dir=self.paths.asset_preparation,
                tile_grid_path=tiles_path,
                buffer_distance=self.config["input"]["buffer_distance"],
                segment_length=self.config["input"]["segment_length"],
            )
            self.config["input"]["area_polygon"] = asset_results["buffer_path"]
            self.config["input"]["tile_grid"] = asset_results["intersecting_tiles_path"]

            logger.debug(f"Asset preparation results: {asset_results}")
            logger.debug(f"Buffer path exists: {os.path.exists(asset_results['buffer_path'])}")
            tiled_buffers = asset_results.get("tiled_buffers")
            if isinstance(tiled_buffers, dict):
                sample = next(iter(tiled_buffers.values()), None)
                logger.debug(f"Tiled buffers: {len(tiled_buffers)} entries; sample: {sample}")
            else:
                logger.debug(f"Tiled buffers path: {tiled_buffers}")
            return asset_results

    def _process_imagery(self, asset_results: Dict[str, Any]) -> None:
        """Process imagery using buffered AOI and tiles."""
        logger.info("Step 3: Processing imagery")
        logger.debug(f"Using buffer path: {asset_results['buffer_path']}")
        tiled_buffers = asset_results["tiled_buffers"]
        if isinstance(tiled_buffers, dict):
            sample = next(iter(tiled_buffers.values()), None)
            logger.debug(f"Using tiled buffers dict with {len(tiled_buffers)} entries; sample: {sample}")
        else:
            logger.debug(f"Using tiled buffers path: {tiled_buffers}")

        preprocessor = ImageryPreprocessor(
            input_dir=self.paths.raw_data,
            output_dir=self.paths.processed_data,
            aoi=asset_results["buffer_path"],
            config=self.config,
            debug_mode=self.debug_mode,
        )
        preprocessor.process(
            tiles_path=asset_results["intersecting_tiles_path"],
            tiled_buffer_path=asset_results["tiled_buffers"],
        )

    def _run_segmentation(self) -> None:
        """Execute the segmentation step."""
        with Timer("Segmentation", module="segmentation"):
            processor = RasterProcessor(
                input_dir=self.paths.processed_data,
                output_dir=self.paths.segmentation,
                config=self.config,
                debug_mode=self.debug_mode,
            )
            processor.setup_directories()
            processor.run(redo=True)

    def _run_predictions(self) -> None:
        """Run model predictions and summaries."""
        with Timer("Model Prediction", module="prediction"):
            predict(
                zonal_stats_dir=os.path.join(self.paths.segmentation, "zonal_stats"),
                predictions_dir=self.paths.predictions,
                model_path=self.config["prediction"]["model_load_path"],
                config=self.config,
            )
            logger.info("Generating classification summary...")
            generate_class_summary(self.paths.predictions)

    def _create_visualizations(self) -> None:
        """Generate visualization outputs for predictions."""
        with Timer("Folium Map Creation", module="visualization"):
            logger.info("Creating interactive Folium map from polygon files...")
            raw_dir = os.path.join(self.paths.predictions, "raw")
            polygon_files = []
            if os.path.exists(raw_dir):
                for ext in ["*_predicted.gpkg"]:
                    import glob

                    polygon_files.extend(glob.glob(os.path.join(raw_dir, ext)))

            if polygon_files:
                logger.info(f"Found {len(polygon_files)} polygon files for visualization")
                map_path = create_classification_map_from_dissolved_gpkg(
                    predictions_dir=self.paths.predictions,
                    map_title="Multi-Class Classification Results (From Dissolved GPKG - No Simplification)",
                )
                if not map_path:
                    logger.info("Dissolved GPKG not found, falling back to regular polygon method")
                    map_path = create_classification_map_from_polygons(
                        predictions_dir=self.paths.predictions,
                        map_title="Multi-Class Classification Results (From Polygons)",
                    )
                if map_path:
                    logger.info(f"Interactive map created from polygons: {map_path}")
                else:
                    logger.warning("Failed to create interactive map from polygons")
                return

            logger.info("No polygon files found, looking for TIFF files...")
            raster_dir = os.path.join(self.paths.predictions, "raster", "multiclass")
            tiff_files = []
            if os.path.exists(raster_dir):
                for ext in ["*.tif", "*.tiff"]:
                    import glob

                    tiff_files.extend(glob.glob(os.path.join(raster_dir, ext)))

            if tiff_files:
                tiff_path = tiff_files[0]
                logger.info(f"Using TIFF file: {tiff_path}")
                aoi_path = os.path.join(self.config["directories"]["base_output_dir"], "asset", "vse_aoi_hv.gpkg")
                if os.path.exists(aoi_path):
                    logger.info(f"Using AOI for clipping: {aoi_path}")
                else:
                    logger.warning(f"AOI file not found: {aoi_path}")
                    aoi_path = None

                map_path = create_classification_map_from_tiff(
                    tiff_path=tiff_path,
                    output_dir=self.paths.predictions,
                    aoi_path=aoi_path,
                    map_title="Multi-Class Classification Results (From TIFF)",
                )
                if map_path:
                    logger.info(f"Interactive map created from TIFF: {map_path}")
                else:
                    logger.warning("Failed to create interactive map from TIFF")
                return

            logger.info("No TIFF files found, using prediction files...")
            map_path = create_classification_map(
                predictions_dir=self.paths.predictions,
                map_title="Multi-Class Classification Results",
            )
            if map_path:
                logger.info(f"Interactive map created from prediction files: {map_path}")
            else:
                logger.warning("Failed to create interactive map from prediction files")

    @timer_decorator(module="pipeline")
    def run_pipeline(self):
        """Run the complete end-to-end classification pipeline."""
        try:
            logger.info("Starting E2E Classification Pipeline")
            tiles_path = self._download_sentinel_data()
            asset_results = self._prepare_assets(tiles_path)
            self._process_imagery(asset_results)
            self._run_segmentation()
            self._run_predictions()
            self._create_visualizations()

            DurationCollector().print_summary()
            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def create_tiles_from_downloaded_scenes(raw_data_dir: str, output_dir: str) -> str:
    """
    Create a vector file of tiles from downloaded Planet scenes.
    
    Args:
        raw_data_dir: Directory containing downloaded Planet scenes
        output_dir: Directory to save the tiles vector file
        
    Returns:
        Path to the created tiles vector file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all downloaded TIFF files
    tiff_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.tif')]
    
    if not tiff_files:
        raise ValueError("No downloaded scenes found in raw_data directory")
    
    # Create a list to store tile geometries and properties
    geometries = []
    properties = []
    
    # Get extent of each scene
    for tiff_file in tiff_files:
        tiff_path = os.path.join(raw_data_dir, tiff_file)
        with rasterio.open(tiff_path) as src:
            # Get the bounds of the raster
            bounds = src.bounds
            # Create a polygon from the bounds
            tile_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            # Extract tile_id from filename (remove .tif extension)
            tile_id = os.path.splitext(tiff_file)[0]
            
            # Add to lists
            geometries.append(tile_geom)
            properties.append({
                'tile_id': tile_id,
                'filename': tiff_file
            })
    
    # Create GeoDataFrame with Web Mercator CRS (EPSG:3857)
    gdf = gpd.GeoDataFrame(
        data=properties,
        geometry=geometries,
        crs="EPSG:3857"  # Explicitly set Web Mercator CRS
    )
    
    # Reproject from Web Mercator (3857) to WGS84 (4326)
    logger.info("Reprojecting tiles from EPSG:3857 to EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    
    # Set tile_id as index for easier access
    gdf = gdf.set_index('tile_id')
    
    # Save to file
    tiles_path = os.path.join(output_dir, 'tiles.geojson')
    gdf.to_file(tiles_path, driver='GeoJSON')
    
    logger.info(f"Created tiles vector file with {len(tiff_files)} tiles: {tiles_path}")
    return tiles_path


