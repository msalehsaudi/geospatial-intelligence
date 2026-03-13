#!/usr/bin/env python3
"""
Folium Map Visualization Module for Multi-Class Classification Results

This module creates interactive HTML maps showing the classification results
with different colors for each class and confidence-based styling.
"""

import os
import glob
import geopandas as gpd
import pandas as pd
import folium
from loguru import logger
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
from shapely.ops import unary_union
import json

# Class mapping and colors
CLASS_NAMES = {
    0: 'grassland',
    1: 'tree', 
    2: 'urban'
}

CLASS_COLORS = {
    'grassland': '#2ca02c',  # Green
    'tree': '#1f77b4',       # Blue  
    'urban': '#d62728'        # Red
}

CLASS_DISPLAY_NAMES = {
    'grassland': 'Grassland',
    'tree': 'Tree',
    'urban': 'Urban'
}

def load_prediction_data(predictions_dir: str) -> Optional[gpd.GeoDataFrame]:
    """
    Load all prediction data from the raw predictions directory.
    
    Args:
        predictions_dir: Local directory containing prediction results
        
    Returns:
        Combined GeoDataFrame with all predictions, or None if no data found
    """
    logger.info("Loading prediction data for visualization...")
    
    if isinstance(predictions_dir, str) and predictions_dir.startswith("s3://"):
        raise ValueError("S3 prediction directories are no longer supported. Please use a local directory.")
    raw_dir = os.path.join(predictions_dir, "raw")
    if not os.path.exists(raw_dir):
        logger.warning(f"Raw predictions directory not found: {raw_dir}")
        return None
    
    prediction_files = glob.glob(os.path.join(raw_dir, "*_predicted.gpkg"))
    logger.info(f"Found {len(prediction_files)} prediction files locally")
    
    if not prediction_files:
        logger.warning("No prediction files found")
        return None
    
    all_predictions = []
    
    for file_path in prediction_files:
        try:
            gdf = gpd.read_file(file_path)
            
            if not gdf.empty:
                # Ensure we have the required columns
                required_cols = ['predicted_class', 'class_name', 'confidence', 'geometry']
                missing_cols = [col for col in required_cols if col not in gdf.columns]
                if missing_cols:
                    logger.warning(f"Missing columns in {os.path.basename(file_path)}: {missing_cols}")
                    continue
                
                all_predictions.append(gdf)
                logger.info(f"Loaded {len(gdf)} predictions from {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    if not all_predictions:
        logger.warning("No valid prediction data found")
        return None
    
    # Combine all predictions
    combined_gdf = pd.concat(all_predictions, ignore_index=True)
    
    # Ensure CRS is WGS84 for Folium
    if combined_gdf.crs is None:
        combined_gdf = combined_gdf.set_crs('EPSG:4326')
    elif combined_gdf.crs != 'EPSG:4326':
        combined_gdf = combined_gdf.to_crs('EPSG:4326')
    
    # Note: Overlap resolution happens after dissolving by class_name
    # The dissolve operation groups polygons by class, then we apply priority-based
    # difference in create_classification_map_from_polygons (tree > grassland > urban)
    
    logger.info(f"Final combined predictions: {len(combined_gdf)}")
    return combined_gdf

def create_folium_map(predictions_gdf: gpd.GeoDataFrame, 
                     output_path: str,
                     map_title: str = "Multi-Class Classification Results",
                     summary_panel_html: str = None) -> str:
    """
    Create a Folium map visualization of the classification results.
    
    Args:
        predictions_gdf: GeoDataFrame with prediction results
        output_path: Path to save the HTML file
        map_title: Title for the map
        
    Returns:
        Path to the saved HTML file
    """
    logger.info("Creating Folium map visualization...")
    
    if predictions_gdf is None or predictions_gdf.empty:
        logger.error("No prediction data provided for visualization")
        return None
    
    # Calculate map center from data bounds
    # Ensure data is in EPSG:4326 for proper lat/lon coordinates
    if predictions_gdf.crs != 'EPSG:4326':
        predictions_gdf = predictions_gdf.to_crs('EPSG:4326')
    
    bounds = predictions_gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap',
        control_scale=True,
        prefer_canvas=False
    )
    
    # Add tile layers
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark Matter').add_to(m)
    
    # Add satellite imagery
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False
    ).add_to(m)
    
    # Create feature groups for each class
    feature_groups = {}
    for class_name in CLASS_NAMES.values():
        display_name = CLASS_DISPLAY_NAMES.get(class_name, class_name.title())
        feature_groups[class_name] = folium.FeatureGroup(name=display_name)

    # Prefer GeoJSON layers to avoid Leaflet vertex simplification and preserve holes
    try:
        for class_name in CLASS_NAMES.values():
            class_color = CLASS_COLORS.get(class_name, '#000000')
            class_df = predictions_gdf[predictions_gdf['class_name'] == class_name]
            if class_df.empty:
                continue

            # Ensure geometries exist (avoid altering vertices)
            class_df = class_df[class_df.geometry.notnull()]

            # Build popup template using GeoJsonPopup for basic fields
            popup = None
            popup_fields = []
            if 'predicted_class' in class_df.columns:
                popup_fields.append('predicted_class')
            if 'class_name' in class_df.columns:
                popup_fields.append('class_name')
            if 'confidence' in class_df.columns:
                popup_fields.append('confidence')
            if 'area_sq_km' in class_df.columns:
                popup_fields.append('area_sq_km')

            if popup_fields:
                try:
                    popup = folium.features.GeoJsonPopup(
                        fields=popup_fields,
                        aliases=[
                            'Class ID' if f == 'predicted_class' else (
                                'Class' if f == 'class_name' else (
                                    'Confidence' if f == 'confidence' else (
                                        'Area (km²)' if f == 'area_sq_km' else f
                                    )
                                )
                            ) for f in popup_fields
                        ],
                        localize=True,
                        labels=True,
                        style="background: white; border: 1px solid #ccc; border-radius: 4px; padding: 6px;"
                    )
                except Exception:
                    popup = None

            gj = folium.GeoJson(
                data=json.loads(class_df.to_json()),
                name=CLASS_DISPLAY_NAMES.get(class_name, class_name.title()),
                style_function=lambda _,
                                  color=class_color: {
                    'color': color,
                    'fillColor': color,
                    'fillOpacity': 0.3,
                    'weight': 1
                },
                smooth_factor=0,
                precision=10,
                embed=False,
                zoom_on_click=False,
                tooltip=None,
                popup=popup
            )
            gj.add_to(feature_groups[class_name])

        # Add feature groups to map
        for class_name, fg in feature_groups.items():
            fg.add_to(m)

    except Exception:
        # Fallback: per-feature Polygon rendering (will try to disable smoothing)
        # Add features to the map
        for idx, row in predictions_gdf.iterrows():
            try:
                # Get geometry and properties
                geom = row.geometry
                class_name = row.get('class_name', None)
                confidence = row.get('confidence', None)
                predicted_class = row.get('predicted_class', None)
                
                if geom is None or class_name not in CLASS_COLORS:
                    continue
                
                # Get color and use transparent opacity
                color = CLASS_COLORS.get(class_name, '#000000')
                opacity = 0.3  # Fixed transparent opacity for all features
                
                # Create popup content (robust to missing fields)
                rows_html = [
                    f"<tr><td><strong>Class:</strong></td><td>{class_name}</td></tr>"
                ]
                if predicted_class is not None:
                    rows_html.append(f"<tr><td><strong>Class ID:</strong></td><td>{predicted_class}</td></tr>")
                if confidence is not None:
                    try:
                        rows_html.append(f"<tr><td><strong>Confidence:</strong></td><td>{float(confidence):.3f}</td></tr>")
                    except Exception:
                        rows_html.append(f"<tr><td><strong>Confidence:</strong></td><td>{confidence}</td></tr>")
                # Include area if present
                if 'area_sq_km' in predictions_gdf.columns:
                    try:
                        rows_html.append(f"<tr><td><strong>Area (km²):</strong></td><td>{float(row['area_sq_km']):.3f}</td></tr>")
                    except Exception:
                        pass
                popup_content = f"""
                <div style="font-family: Arial, sans-serif; min-width: 200px;">
                    <h4 style="margin: 0 0 10px 0; color: {color};">{CLASS_DISPLAY_NAMES.get(class_name, (class_name or '').title())}</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        {''.join(rows_html)}
                    </table>
                </div>
                """
                
                # Add geometry to map based on type
                if geom.geom_type == 'Polygon':
                    # Exterior ring
                    if hasattr(geom, 'exterior') and geom.exterior is not None:
                        exterior_coords = [[y, x] for x, y in geom.exterior.coords]
                    else:
                        exterior_coords = [[y, x] for x, y in geom.coords]

                    # Interior rings (holes)
                    holes_coords = []
                    if hasattr(geom, 'interiors') and geom.interiors:
                        for ring in geom.interiors:
                            holes_coords.append([[y, x] for x, y in ring.coords])

                    folium.Polygon(
                        locations=exterior_coords,
                        holes=holes_coords if holes_coords else None,
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=opacity,
                        weight=1,
                        smooth_factor=0,
                        popup=folium.Popup(popup_content, max_width=300)
                    ).add_to(feature_groups[class_name])
                    
                elif geom.geom_type == 'MultiPolygon':
                    # Handle MultiPolygon with holes and no smoothing
                    for poly in geom.geoms:
                        exterior_coords = [[y, x] for x, y in poly.exterior.coords]
                        holes_coords = []
                        if poly.interiors:
                            for ring in poly.interiors:
                                holes_coords.append([[y, x] for x, y in ring.coords])
                        folium.Polygon(
                            locations=exterior_coords,
                            holes=holes_coords if holes_coords else None,
                            color=color,
                            fill=True,
                            fillColor=color,
                            fillOpacity=opacity,
                            weight=1,
                            smooth_factor=0,
                            popup=folium.Popup(popup_content, max_width=300)
                        ).add_to(feature_groups[class_name])
            
            except Exception as e:
                logger.warning(f"Error processing feature {idx}: {e}")
                continue
    
        # Add feature groups to map (fallback path)
        for class_name, fg in feature_groups.items():
            fg.add_to(m)
    
    # Add properly stacked legend (no overlay)
    legend_html = """
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; opacity: 0.9;">
    <h4 style="margin: 0 0 10px 0;">Classification Legend</h4>
    """
    
    # Add class legend
    for class_name, color in CLASS_COLORS.items():
        display_name = CLASS_DISPLAY_NAMES.get(class_name, class_name.title())
        legend_html += f"""
        <p style="margin: 5px 0;">
            <span style="color: {color}; font-weight: bold;">■</span> {display_name}
        </p>
        """
    
    legend_html += """
    <hr style="margin: 10px 0; border: 1px solid #ccc;">
    <p style="margin: 5px 0; font-size: 12px; color: #666;">
        All features have transparent opacity (30%)
    </p>
    </div>
    """
    
    # Add legend to map
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control in a separate position to avoid overlay
    folium.LayerControl(
        position='bottomright',
        collapsed=False,
        overlay=True
    ).add_to(m)
    
    # Optional summary panel at bottom
    if summary_panel_html:
        m.get_root().html.add_child(folium.Element(summary_panel_html))

    if isinstance(output_path, str) and output_path.startswith("s3://"):
        raise ValueError("S3 output paths are no longer supported. Please use a local HTML path.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        logger.info(f"Saving Folium map to: {output_path}")
        m.save(output_path)
        logger.info(f"Folium map saved successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving Folium map: {e}")
        raise
    
    return output_path

def generate_visualization_summary(predictions_gdf: gpd.GeoDataFrame) -> Dict:
    """
    Generate a summary of the visualization data.
    
    Args:
        predictions_gdf: GeoDataFrame with prediction results
        
    Returns:
        Dictionary with summary statistics
    """
    if predictions_gdf is None or predictions_gdf.empty:
        return {}
    
    summary = {
        'total_features': len(predictions_gdf),
        'classes': {},
        'confidence_stats': {
            'mean': predictions_gdf['confidence'].mean(),
            'min': predictions_gdf['confidence'].min(),
            'max': predictions_gdf['confidence'].max(),
            'std': predictions_gdf['confidence'].std()
        }
    }
    
    # Class statistics
    for class_name in CLASS_NAMES.values():
        class_mask = predictions_gdf['class_name'] == class_name
        class_count = class_mask.sum()
        class_confidence = predictions_gdf.loc[class_mask, 'confidence']
        
        summary['classes'][class_name] = {
            'count': class_count,
            'percentage': (class_count / len(predictions_gdf)) * 100,
            'mean_confidence': class_confidence.mean() if len(class_confidence) > 0 else 0,
            'min_confidence': class_confidence.min() if len(class_confidence) > 0 else 0,
            'max_confidence': class_confidence.max() if len(class_confidence) > 0 else 0
        }
    
    return summary

def create_classification_map(predictions_dir: str, 
                           output_dir: str = None,
                           map_title: str = "Multi-Class Classification Results") -> Optional[str]:
    """
    Main function to create a Folium map visualization of classification results.
    
    Args:
        predictions_dir: Directory containing prediction results
        output_dir: Directory to save the HTML file (defaults to predictions_dir)
        map_title: Title for the map
        
    Returns:
        Path to the saved HTML file, or None if failed
    """
    logger.info("Starting Folium map creation (Step 6 - Optional)")
    
    try:
        # Set output directory
        if output_dir is None:
            output_dir = predictions_dir
        
        # Load prediction data
        predictions_gdf = load_prediction_data(predictions_dir)
        if predictions_gdf is None:
            logger.error("No prediction data found for visualization")
            return None
        
        # Generate summary
        summary = generate_visualization_summary(predictions_gdf)
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"classification_map_{timestamp}.html"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create the map
        map_path = create_folium_map(predictions_gdf, output_path, map_title)
        
        if map_path:
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("FOLIUM MAP CREATION SUMMARY")
            logger.info("="*60)
            logger.info(f"Output file: {map_path}")
            logger.info(f"Total features: {summary.get('total_features', 0)}")
            logger.info(f"Mean confidence: {summary.get('confidence_stats', {}).get('mean', 0):.3f}")
            logger.info("\nClass Distribution:")
            for class_name, stats in summary.get('classes', {}).items():
                display_name = CLASS_DISPLAY_NAMES.get(class_name, class_name.title())
                logger.info(f"  {display_name}: {stats['count']} features ({stats['percentage']:.1f}%)")
                logger.info(f"    Mean confidence: {stats['mean_confidence']:.3f}")
            logger.info("="*60)
        
        return map_path
        
    except Exception as e:
        logger.error(f"Error creating Folium map: {e}")
        return None

def create_classification_map_from_tiff(tiff_path: str,
                                     output_dir: str = None,
                                     aoi_path: str = None,
                                     map_title: str = "Multi-Class Classification Results") -> Optional[str]:
    """
    Create a Folium map directly from a polygonized TIFF file.
    
    Args:
        tiff_path: Path to the classification TIFF file
        output_dir: Directory to save the HTML file
        aoi_path: Path to AOI boundary file for clipping (optional)
        map_title: Title for the map
        
    Returns:
        Path to the saved HTML file, or None if failed
    """
    logger.info("Starting Folium map creation from TIFF (Step 6 - Optional)")
    
    try:
        # Import the polygonizer
        from treelance_sentinel.tiff_polygonizer import polygonize_tiff
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(tiff_path)
        
        # Polygonize the TIFF file
        logger.info(f"Polygonizing TIFF file: {tiff_path}")
        predictions_gdf = polygonize_tiff(tiff_path, aoi_path=aoi_path)
        
        if predictions_gdf is None:
            logger.error("Failed to polygonize TIFF file")
            return None
        
        # Generate summary
        summary = generate_visualization_summary(predictions_gdf)
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"classification_map_{timestamp}.html"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create the map
        map_path = create_folium_map(predictions_gdf, output_path, map_title)
        
        if map_path:
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("FOLIUM MAP CREATION SUMMARY (FROM TIFF)")
            logger.info("="*60)
            logger.info(f"Input TIFF: {tiff_path}")
            logger.info(f"Output file: {map_path}")
            logger.info(f"Total features: {summary.get('total_features', 0)}")
            logger.info(f"Mean confidence: {summary.get('confidence_stats', {}).get('mean', 0):.3f}")
            logger.info("\nClass Distribution:")
            for class_name, stats in summary.get('classes', {}).items():
                display_name = CLASS_DISPLAY_NAMES.get(class_name, class_name.title())
                logger.info(f"  {display_name}: {stats['count']} features ({stats['percentage']:.1f}%)")
                logger.info(f"    Mean confidence: {stats['mean_confidence']:.3f}")
            logger.info("="*60)
        
        return map_path
        
    except Exception as e:
        logger.error(f"Error creating Folium map from TIFF: {e}")
        return None

def create_classification_map_from_dissolved_gpkg(predictions_dir: str,
                                                output_dir: str = None,
                                                map_title: str = "Multi-Class Classification Results") -> Optional[str]:
    """
    Create a Folium map directly from the dissolved_predictions_*.gpkg file.
    
    Args:
        predictions_dir: Directory containing prediction results
        output_dir: Directory to save the HTML file
        map_title: Title for the map
        
    Returns:
        Path to the saved HTML file, or None if failed
    """
    logger.info("Starting Folium map creation from dissolved GPKG file")
    
    try:
        # Set output directory
        if output_dir is None:
            output_dir = predictions_dir
        
        if isinstance(predictions_dir, str) and predictions_dir.startswith("s3://"):
            raise ValueError("S3 prediction directories are no longer supported. Please use a local directory.")
        polygons_dir = os.path.join(predictions_dir, 'polygons')
        dissolved_files = glob.glob(os.path.join(polygons_dir, "dissolved_predictions_*.gpkg"))
        # Sort by modification time (most recent first)
        if dissolved_files:
            dissolved_files.sort(key=os.path.getctime, reverse=True)
        
        if not dissolved_files:
            logger.warning("No dissolved GPKG files found in polygons directory")
            return None
        
        # Use the most recent dissolved file
        dissolved_path = dissolved_files[0]
        logger.info(f"Using dissolved GPKG file: {os.path.basename(dissolved_path)}")
        
        # Load the dissolved polygons
        dissolved = gpd.read_file(dissolved_path, layer='dissolved')
        
        if dissolved.empty:
            logger.error("Dissolved GPKG file is empty")
            return None
        
        # Ensure CRS is WGS84 for Folium
        if dissolved.crs is None:
            dissolved = dissolved.set_crs('EPSG:4326')
        elif dissolved.crs != 'EPSG:4326':
            dissolved = dissolved.to_crs('EPSG:4326')
        
        logger.info(f"Loaded {len(dissolved)} dissolved polygons")
        
        # Build area summary panel HTML
        total_area = dissolved['area_sq_km'].sum(skipna=True)
        rows = []
        for cls in ['tree', 'urban', 'grassland']:
            cls_area = dissolved.loc[dissolved['class_name'] == cls, 'area_sq_km']
            cls_area_val = float(cls_area.iloc[0]) if len(cls_area) > 0 and pd.notnull(cls_area.iloc[0]) else 0.0
            pct = (cls_area_val / total_area * 100.0) if total_area and total_area > 0 else 0.0
            rows.append((cls, cls_area_val, pct))

        # Create HTML panel (fixed bottom)
        panel_rows_html = "".join([
            f"<tr><td style='padding:4px 8px; text-align:left;'>{CLASS_DISPLAY_NAMES.get(cls, cls.title())}</td>"
            f"<td style='padding:4px 8px; text-align:right;'>{area_km2:.2f} km²</td>"
            f"<td style='padding:4px 8px; text-align:right;'>{pct:.1f}%</td></tr>"
            for cls, area_km2, pct in rows
        ])
        summary_panel_html = f"""
        <div style="position: fixed; bottom: 10px; left: 10px; width: 280px; background: rgba(255,255,255,0.95); border: 1px solid #ccc; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); z-index: 9999; font-family: Arial, sans-serif;">
            <div style="padding: 8px 12px;">
                <div style="display:flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600;">Area Summary</div>
                    <div style="color:#666;">Total: {total_area:.2f} km²</div>
                </div>
                <table style="width: 100%; border-collapse: collapse; margin-top:6px; font-size: 12px;">
                    <thead>
                        <tr style="text-align:left; border-bottom:1px solid #eee;">
                            <th style="padding:4px 6px;">Class</th>
                            <th style="padding:4px 6px; text-align:right;">Area</th>
                            <th style="padding:4px 6px; text-align:right;">%</th>
                        </tr>
                    </thead>
                    <tbody>
                        {panel_rows_html}
                    </tbody>
                </table>
            </div>
        </div>
        """

        # Create output filename with timestamp for HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"classification_map_dissolved_{timestamp}.html"
        output_path = os.path.join(output_dir, output_filename)

        # Create the map using dissolved geometries and include summary panel
        map_path = create_folium_map(dissolved, output_path, map_title, summary_panel_html=summary_panel_html)
        
        if map_path:
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("FOLIUM MAP CREATION SUMMARY (FROM DISSOLVED GPKG)")
            logger.info("="*60)
            logger.info(f"Input file: {os.path.basename(dissolved_path)}")
            logger.info(f"Output file: {map_path}")
            logger.info(f"Total features: {len(dissolved)}")
            logger.info("\nClass Distribution:")
            for class_name in dissolved['class_name'].unique():
                class_count = len(dissolved[dissolved['class_name'] == class_name])
                display_name = CLASS_DISPLAY_NAMES.get(class_name, class_name.title())
                logger.info(f"  {display_name}: {class_count} features")
            logger.info("="*60)
        
        return map_path
        
    except Exception as e:
        logger.error(f"Error creating Folium map from dissolved GPKG: {e}")
        return None

def create_classification_map_from_polygons(predictions_dir: str,
                                         output_dir: str = None,
                                         map_title: str = "Multi-Class Classification Results") -> Optional[str]:
    """
    Create a Folium map directly from GPKG polygon files (preferred method).
    
    Args:
        predictions_dir: Directory containing prediction results
        output_dir: Directory to save the HTML file
        map_title: Title for the map
        
    Returns:
        Path to the saved HTML file, or None if failed
    """
    logger.info("Starting Folium map creation from polygon files (Step 6 - Preferred)")
    
    try:
        # Set output directory
        if output_dir is None:
            output_dir = predictions_dir
        
        if isinstance(predictions_dir, str) and predictions_dir.startswith("s3://"):
            raise ValueError("S3 prediction directories are no longer supported. Please use a local directory.")
        raw_dir = os.path.join(predictions_dir, "raw")
        viz_files = glob.glob(os.path.join(raw_dir, "*_predicted.gpkg"))
        
        if not viz_files:
            logger.warning("No predicted GPKG files found in raw directory")
            return None
        
        # Load all polygon files
        all_predictions = []
        for file_path in viz_files:
            try:
                gdf = gpd.read_file(file_path)

                if not gdf.empty:
                    all_predictions.append(gdf)
                    logger.info(f"Loaded {len(gdf)} predictions from {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not all_predictions:
            logger.error("No valid prediction data found in polygon files")
            return None
        
        # Combine all predictions
        predictions_gdf = pd.concat(all_predictions, ignore_index=True)
        
        # Note: Do not sample before dissolve. We want full coverage in dissolved output and map.

        # Generate summary from the full (pre-dissolve) dataset for logging
        summary = generate_visualization_summary(predictions_gdf)

        # Ensure CRS is set for area calculation and dissolve
        if predictions_gdf.crs is None:
            predictions_gdf = predictions_gdf.set_crs('EPSG:4326')

        # Dissolve polygons by class_name (union per class)
        try:
            # Fix invalid geometries before dissolve
            predictions_gdf['geometry'] = predictions_gdf.geometry.buffer(0)
        except Exception:
            pass

        dissolved = predictions_gdf.dissolve(by='class_name', as_index=False)

        # Enforce non-overlapping areas between classes for clearer visualization
        # Priority order: tree > grassland > urban (trees have highest priority)
        try:
            priority = ['tree', 'grassland', 'urban']
            occupied = None
            adjusted_rows = []
            for cls in priority:
                cls_row = dissolved[dissolved['class_name'] == cls]
                if cls_row.empty:
                    continue
                geom = cls_row.iloc[0].geometry
                if occupied is not None and not occupied.is_empty:
                    try:
                        geom = geom.difference(occupied)
                    except Exception:
                        pass
                adjusted_rows.append({'class_name': cls, 'geometry': geom})
                # Update occupied with the original (pre-difference) class geometry so higher priority remains intact
                occupied = unary_union([g for g in [occupied, cls_row.iloc[0].geometry] if g is not None])
            if adjusted_rows:
                dissolved = gpd.GeoDataFrame(adjusted_rows, geometry='geometry', crs=predictions_gdf.crs)
        except Exception as e:
            logger.warning(f"Failed to enforce non-overlapping classes: {e}")

        # Compute area per class in km^2 (project to EPSG:3857 for meters)
        try:
            dissolved_m = dissolved.to_crs('EPSG:3857')
            dissolved['area_sq_km'] = dissolved_m.geometry.area / 1_000_000.0
        except Exception as e:
            logger.warning(f"Area computation failed on dissolve: {e}")
            dissolved['area_sq_km'] = np.nan

        # Save dissolved to GPKG under predictions/polygons
        polygons_dir = os.path.join(predictions_dir, 'polygons')
        os.makedirs(polygons_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dissolved_path = os.path.join(polygons_dir, f"dissolved_predictions_{timestamp}.gpkg")
        try:
            # Save with layer name 'dissolved'
            dissolved.to_file(dissolved_path, driver='GPKG', layer='dissolved')
            logger.info(f"Saved dissolved polygons to: {dissolved_path}")
        except Exception as e:
            logger.warning(f"Failed to save dissolved GPKG: {e}")

        # Build area summary panel HTML
        total_area = dissolved['area_sq_km'].sum(skipna=True)
        rows = []
        for cls in ['tree', 'urban', 'grassland']:
            cls_area = dissolved.loc[dissolved['class_name'] == cls, 'area_sq_km']
            cls_area_val = float(cls_area.iloc[0]) if len(cls_area) > 0 and pd.notnull(cls_area.iloc[0]) else 0.0
            pct = (cls_area_val / total_area * 100.0) if total_area and total_area > 0 else 0.0
            rows.append((cls, cls_area_val, pct))

        # Create HTML panel (fixed bottom)
        panel_rows_html = "".join([
            f"<tr><td style='padding:4px 8px; text-align:left;'>{CLASS_DISPLAY_NAMES.get(cls, cls.title())}</td>"
            f"<td style='padding:4px 8px; text-align:right;'>{area_km2:.2f} km²</td>"
            f"<td style='padding:4px 8px; text-align:right;'>{pct:.1f}%</td></tr>"
            for cls, area_km2, pct in rows
        ])
        summary_panel_html = f"""
        <div style="position: fixed; bottom: 10px; left: 10px; width: 280px; background: rgba(255,255,255,0.95); border: 1px solid #ccc; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); z-index: 9999; font-family: Arial, sans-serif;">
            <div style="padding: 8px 12px;">
                <div style="display:flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600;">Area Summary</div>
                    <div style="color:#666;">Total: {total_area:.2f} km²</div>
                </div>
                <table style="width: 100%; border-collapse: collapse; margin-top:6px; font-size: 12px;">
                    <thead>
                        <tr style="text-align:left; border-bottom:1px solid #eee;">
                            <th style="padding:4px 6px;">Class</th>
                            <th style="padding:4px 6px; text-align:right;">Area</th>
                            <th style="padding:4px 6px; text-align:right;">%</th>
                        </tr>
                    </thead>
                    <tbody>
                        {panel_rows_html}
                    </tbody>
                </table>
            </div>
        </div>
        """

        # Create output filename with timestamp for HTML
        output_filename = f"classification_map_polygons_{timestamp}.html"
        output_path = os.path.join(output_dir, output_filename)

        # Create the map using dissolved geometries and include summary panel
        map_path = create_folium_map(dissolved, output_path, map_title, summary_panel_html=summary_panel_html)
        
        if map_path:
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("FOLIUM MAP CREATION SUMMARY (FROM POLYGONS)")
            logger.info("="*60)
            logger.info(f"Input files: {len(viz_files)} GPKG files")
            logger.info(f"Output file: {map_path}")
            logger.info(f"Total features: {summary.get('total_features', 0)}")
            logger.info(f"Mean confidence: {summary.get('confidence_stats', {}).get('mean', 0):.3f}")
            logger.info("\nClass Distribution:")
            for class_name, stats in summary.get('classes', {}).items():
                display_name = CLASS_DISPLAY_NAMES.get(class_name, class_name.title())
                logger.info(f"  {display_name}: {stats['count']} features ({stats['percentage']:.1f}%)")
                logger.info(f"    Mean confidence: {stats['mean_confidence']:.3f}")
            logger.info("="*60)
        
        return map_path
        
    except Exception as e:
        logger.error(f"Error creating Folium map from polygons: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Folium map visualization of classification results')
    parser.add_argument('--predictions_dir', required=True, help='Directory containing prediction results')
    parser.add_argument('--output_dir', help='Directory to save the HTML file')
    parser.add_argument('--title', default='Multi-Class Classification Results', help='Map title')
    
    args = parser.parse_args()
    
    result = create_classification_map(args.predictions_dir, args.output_dir, args.title)
    if result:
        print(f"Map created successfully: {result}")
    else:
        print("Failed to create map")
