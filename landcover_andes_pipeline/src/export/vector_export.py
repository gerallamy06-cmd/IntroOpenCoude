"""
Módulo Exportación Vectorial: GPKG/Shapefile transiciones y hotspots
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping, box, Polygon
from rasterio.features import shapes
import rasterio
from rasterio.transform import Affine
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Nombres de clases MapBiomas
CLASS_NAMES = {
    1: "Bosque", 2: "Pastizales", 3: "Cultivos",
    4: "Urbano", 5: "Agua", 6: "Suelo desnudo", 7: "Humedales"
}

# Transiciones de interés
TRANSITION_LABELS = {
    12: "Bosque → Pastizales", 13: "Bosque → Cultivos", 14: "Bosque → Urbano",
    21: "Pastizales → Bosque", 23: "Pastizales → Cultivos", 24: "Pastizales → Urbano", 25: "Pastizales → Agua", 27: "Pastizales → Humedales",
    31: "Cultivos → Bosque", 32: "Cultivos → Pastizales", 34: "Cultivos → Urbano",
    41: "Urbano → Bosque", 42: "Urbano → Pastizales", 43: "Urbano → Cultivos",
    52: "Agua → Pastizales", 57: "Agua → Humedales",
    72: "Humedales → Pastizales", 75: "Humedales → Agua",
}


def raster_to_polygons(
    raster_path: str,
    min_area_ha: float = 0.01,
    simplify_tolerance: float = 0.0,
    connect_8way: bool = True
) -> gpd.GeoDataFrame:
    """
    Convierte raster clasificado a polígonos vectoriales.
    
    Args:
        raster_path: Ruta al GeoTIFF
        min_area_ha: Área mínima en hectáreas para mantener polígono
        simplify_tolerance: Tolerancia para simplificar geometrías (grados)
        connect_8way: Conectividad 8-direcciones vs 4-direcciones
    
    Returns:
        GeoDataFrame con polígonos y atributos de clase
    """
    logger.info(f"Vectorizando raster: {raster_path}")
    
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        
        # Calcular área de píxel en hectáreas
        pixel_width = abs(transform[0])
        pixel_height = abs(transform[4])
        # Aproximación para CRS geográfico (EPSG:4326)
        if crs and crs.to_epsg() == 4326:
            pixel_area_ha = (pixel_width * 111000 * 0.997) * (pixel_height * 111000) / 10000
        else:
            pixel_area_ha = (pixel_width * pixel_height) / 10000
        
        min_pixels = max(1, int(min_area_ha / pixel_area_ha))
        
        # Generar formas
        results = []
        for geom, value in shapes(
            data,
            mask=(data != nodata) if nodata is not None else None,
            transform=transform,
            connectivity=8 if connect_8way else 4
        ):
            if value == 0 or value == nodata:
                continue
            
            # Calcular área aproximada en píxeles
            # Usar bounding box para estimación rápida
            poly = Polygon(geom['coordinates'][0])
            estimated_pixels = poly.area / (pixel_width * pixel_height)
            
            if estimated_pixels >= min_pixels:
                results.append({
                    'geometry': poly,
                    'class_code': int(value),
                    'class_name': CLASS_NAMES.get(int(value), f"Class_{value}"),
                    'pixel_count': int(round(estimated_pixels)),
                    'area_ha': round(estimated_pixels * pixel_area_ha, 2)
                })
    
    if not results:
        logger.warning("No se generaron polígonos")
        return gpd.GeoDataFrame(columns=['class_code', 'class_name', 'pixel_count', 'area_ha', 'geometry'], crs=crs)
    
    gdf = gpd.GeoDataFrame(results, crs=crs)
    
    # Simplificar si se solicita
    if simplify_tolerance > 0:
        gdf['geometry'] = gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)
        logger.info(f"Geometrías simplificadas (tolerancia: {simplify_tolerance})")
    
    # Disolver por clase para reducir features
    gdf_dissolved = gdf.dissolve(by='class_code', aggfunc='sum').reset_index()
    
    logger.info(f"Vectorización completada: {len(gdf_dissolved)} polígonos por clase")
    return gdf_dissolved


def transition_raster_to_polygons(
    transition_raster: np.ndarray,
    transform: Affine,
    crs,
    min_area_ha: float = 0.01,
    target_transitions: List[int] = None
) -> gpd.GeoDataFrame:
    """
    Convierte raster de transiciones (códigos from*10+to) a polígonos.
    """
    logger.info("Vectorizando raster de transiciones...")
    
    # Calcular área de píxel
    pixel_width = abs(transform[0])
    pixel_height = abs(transform[4])
    if crs and crs.to_epsg() == 4326:
        pixel_area_ha = (pixel_width * 111000 * 0.997) * (pixel_height * 111000) / 10000
    else:
        pixel_area_ha = (pixel_width * pixel_height) / 10000
    
    min_pixels = max(1, int(min_area_ha / pixel_area_ha))
    
    # Filtrar transiciones objetivo
    if target_transitions is None:
        target_transitions = list(TRANSITION_LABELS.keys())
    
    results = []
    unique_codes = np.unique(transition_raster)
    
    for code in unique_codes:
        if code == 0 or code not in target_transitions:
            continue
        
        # Máscara para esta transición
        mask = (transition_raster == code)
        
        # Vectorizar
        for geom, value in shapes(
            mask.astype(np.uint8),
            mask=mask,
            transform=transform,
            connectivity=8
        ):
            if value == 1:  # True
                poly = Polygon(geom['coordinates'][0])
                estimated_pixels = poly.area / (pixel_width * pixel_height)
                
                if estimated_pixels >= min_pixels:
                    from_cls = code // 10
                    to_cls = code % 10
                    
                    results.append({
                        'geometry': poly,
                        'transition_code': int(code),
                        'from_class': from_cls,
                        'from_class_name': CLASS_NAMES.get(from_cls, f"Class_{from_cls}"),
                        'to_class': to_cls,
                        'to_class_name': CLASS_NAMES.get(to_cls, f"Class_{to_cls}"),
                        'transition_name': TRANSITION_LABELS.get(code, f"Transición_{code}"),
                        'pixel_count': int(round(estimated_pixels)),
                        'area_ha': round(estimated_pixels * pixel_area_ha, 2)
                    })
    
    if not results:
        logger.warning("No se generaron polígonos de transición")
        return gpd.GeoDataFrame(columns=['transition_code', 'from_class', 'to_class', 'transition_name', 'pixel_count', 'area_ha', 'geometry'], crs=crs)
    
    gdf = gpd.GeoDataFrame(results, crs=crs)
    
    # Disolver por código de transición
    gdf_dissolved = gdf.dissolve(
        by='transition_code',
        aggfunc={
            'from_class': 'first',
            'from_class_name': 'first',
            'to_class': 'first',
            'to_class_name': 'first',
            'transition_name': 'first',
            'pixel_count': 'sum',
            'area_ha': 'sum'
        }
    ).reset_index()
    
    logger.info(f"Transiciones vectorizadas: {len(gdf_dissolved)} polígonos")
    return gdf_dissolved


def save_transition_vectors(
    change_results: Dict,
    output_dir: str,
    template_stack: Any = None,
    overwrite: bool = False
) -> Dict[str, Path]:
    """
    Guarda vectores de transición para cada par anual y global.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = {}
    
    # Obtener transform y CRS del template
    if template_stack is not None and hasattr(template_stack, 'rio'):
        transform = template_stack.rio.transform()
        crs = template_stack.rio.crs
    else:
        logger.error("Template stack requerido para exportar vectores")
        return saved
    
    # Transiciones anuales
    if "annual" in change_results:
        for pair_key, trans_raster in change_results["annual"]["transition_rasters"].items():
            year_from, year_to = pair_key.split("_")
            
            gdf = transition_raster_to_polygons(
                trans_raster,
                transform,
                crs
            )
            
            if len(gdf) > 0:
                # Añadir metadatos temporales
                gdf['year_from'] = int(year_from)
                gdf['year_to'] = int(year_to)
                gdf['period'] = pair_key
                
                filename = f"transitions_{pair_key}.gpkg"
                filepath = output_dir / filename
                
                if filepath.exists() and not overwrite:
                    logger.warning(f"Existe, omitiendo: {filepath}")
                else:
                    gdf.to_file(filepath, driver="GPKG", layer=f"transitions_{pair_key}")
                    saved[f"transitions_{pair_key}"] = filepath
                    logger.info(f"Guardado: {filepath} ({len(gdf)} features)")
    
    # Transición global
    if "global" in change_results:
        trans_raster = change_results["global"]["transition_raster"]
        year_from, year_to = change_results["global"]["years"]
        
        gdf = transition_raster_to_polygons(
            trans_raster,
            transform,
            crs
        )
        
        if len(gdf) > 0:
            gdf['year_from'] = year_from
            gdf['year_to'] = year_to
            gdf['period'] = f"{year_from}_{year_to}"
            
            filename = "transitions_global.gpkg"
            filepath = output_dir / filename
            
            if filepath.exists() and not overwrite:
                logger.warning(f"Existe, omitiendo: {filepath}")
            else:
                gdf.to_file(filepath, driver="GPKG", layer="transitions_global")
                saved["transitions_global"] = filepath
                logger.info(f"Guardado global: {filepath} ({len(gdf)} features)")
    
    return saved


def save_hotspots_vectors(
    hotspots_results: Dict,
    output_dir: str,
    overwrite: bool = False
) -> Dict[str, Path]:
    """
    Guarda resultados de hotspots (puntos Gi*) como GPKG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = {}
    
    if "hotspots_gdf" not in hotspots_results:
        logger.warning("No hay hotspots_gdf en resultados")
        return saved
    
    gdf = hotspots_results["hotspots_gdf"]
    
    # Añadir metadatos globales
    gdf.attrs['analysis_type'] = 'Getis_Ord_GiStar'
    gdf.attrs['generated_at'] = datetime.now().isoformat()
    gdf.attrs['config'] = json.dumps(hotspots_results.get('summary', {}).get('config', {}), default=str)
    
    # Guardar todos los puntos
    filepath_all = output_dir / "hotspots_all.gpkg"
    if filepath_all.exists() and not overwrite:
        logger.warning(f"Existe, omitiendo: {filepath_all}")
    else:
        gdf.to_file(filepath_all, driver="GPKG", layer="hotspots_all")
        saved["hotspots_all"] = filepath_all
        logger.info(f"Hotspots todos guardados: {filepath_all} ({len(gdf)} features)")
    
    # Guardar solo significativos
    sig_gdf = gdf[gdf['significance'] != 0].copy()
    if len(sig_gdf) > 0:
        filepath_sig = output_dir / "hotspots_significant.gpkg"
        if filepath_sig.exists() and not overwrite:
            logger.warning(f"Existe, omitiendo: {filepath_sig}")
        else:
            sig_gdf.to_file(filepath_sig, driver="GPKG", layer="hotspots_significant")
            saved["hotspots_significant"] = filepath_sig
            logger.info(f"Hotspots significativos: {filepath_sig} ({len(sig_gdf)} features)")
    
    # Guardar solo hot spots (deforestación)
    hot_gdf = gdf[gdf['significance'] == 1].copy()
    if len(hot_gdf) > 0:
        filepath_hot = output_dir / "hotspots_deforestation.gpkg"
        if filepath_hot.exists() and not overwrite:
            logger.warning(f"Existe, omitiendo: {filepath_hot}")
        else:
            hot_gdf.to_file(filepath_hot, driver="GPKG", layer="hotspots_deforestation")
            saved["hotspots_deforestation"] = filepath_hot
            logger.info(f"Hot spots (deforestación): {filepath_hot} ({len(hot_gdf)} features)")
    
    # Guardar cold spots (regeneración)
    cold_gdf = gdf[gdf['significance'] == -1].copy()
    if len(cold_gdf) > 0:
        filepath_cold = output_dir / "hotspots_regeneration.gpkg"
        if filepath_cold.exists() and not overwrite:
            logger.warning(f"Existe, omitiendo: {filepath_cold}")
        else:
            cold_gdf.to_file(filepath_cold, driver="GPKG", layer="hotspots_regeneration")
            saved["hotspots_regeneration"] = filepath_cold
            logger.info(f"Cold spots (regeneración): {filepath_cold} ({len(cold_gdf)} features)")
    
    return saved


def save_aoi_vector(
    aoi_gdf: gpd.GeoDataFrame,
    output_dir: str,
    layer_name: str = "aoi_andina",
    overwrite: bool = False
) -> Path:
    """Guarda AOI como GPKG y Shapefile."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GPKG
    gpkg_path = output_dir / f"{layer_name}.gpkg"
    if gpkg_path.exists() and not overwrite:
        logger.warning(f"Existe, omitiendo: {gpkg_path}")
    else:
        aoi_gdf.to_file(gpkg_path, driver="GPKG", layer=layer_name)
        logger.info(f"AOI guardado GPKG: {gpkg_path}")
    
    # Shapefile (para compatibilidad)
    shp_path = output_dir / f"{layer_name}.shp"
    if shp_path.exists() and not overwrite:
        logger.warning(f"Existe, omitiendo: {shp_path}")
    else:
        aoi_gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
        logger.info(f"AOI guardado Shapefile: {shp_path}")
    
    return gpkg_path


def export_transition_matrices_csv(
    change_results: Dict,
    output_dir: str,
    overwrite: bool = False
) -> Dict[str, Path]:
    """
    Exporta matrices de transición a CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = {}
    
    from .change_detection import matrix_to_dataframe
    
    # Anuales
    if "annual" in change_results:
        for pair_key, matrix in change_results["annual"]["matrices"].items():
            df = change_results["annual"]["dataframes"][pair_key]
            filename = f"transition_matrix_{pair_key}.csv"
            filepath = output_dir / filename
            
            if filepath.exists() and not overwrite:
                logger.warning(f"Existe, omitiendo: {filepath}")
            else:
                df.to_csv(filepath, index=False)
                saved[f"matrix_{pair_key}"] = filepath
    
    # Global
    if "global" in change_results:
        df = change_results["global"]["dataframe"]
        filename = "transition_matrix_global.csv"
        filepath = output_dir / filename
        
        if filepath.exists() and not overwrite:
            logger.warning(f"Existe, omitiendo: {filepath}")
        else:
            df.to_csv(filepath, index=False)
            saved["matrix_global"] = filepath
    
    return saved


def export_statistics_csv(
    area_stats: Dict,
    transition_stats: Dict,
    hotspots_stats: Dict = None,
    output_dir: str = "output/stats",
    overwrite: bool = False
) -> Dict[str, Path]:
    """
    Exporta todas las estadísticas a CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = {}
    
    # Area stats
    for key, df in area_stats.items():
        if isinstance(df, pd.DataFrame):
            filename = f"area_{key}.csv"
            filepath = output_dir / filename
            if filepath.exists() and not overwrite:
                continue
            df.to_csv(filepath, index=False)
            saved[f"area_{key}"] = filepath
    
    # Transition stats
    for key, df in transition_stats.items():
        if isinstance(df, pd.DataFrame):
            filename = f"transition_{key}.csv"
            filepath = output_dir / filename
            if filepath.exists() and not overwrite:
                continue
            df.to_csv(filepath, index=False)
            saved[f"transition_{key}"] = filepath
    
    # Hotspots stats
    if hotspots_stats and 'summary' in hotspots_stats:
        summary_df = pd.DataFrame([hotspots_stats['summary']])
        filepath = output_dir / "hotspots_summary.csv"
        if not (filepath.exists() and not overwrite):
            summary_df.to_csv(filepath, index=False)
            saved["hotspots_summary"] = filepath
        
        if 'by_transition' in hotspots_stats['summary']:
            trans_df = pd.DataFrame(hotspots_stats['summary']['by_transition'])
            filepath = output_dir / "hotspots_by_transition.csv"
            if not (filepath.exists() and not overwrite):
                trans_df.to_csv(filepath, index=False)
                saved["hotspots_by_transition"] = filepath
    
    return saved


def create_consolidated_geopackage(
    vectors_dict: Dict[str, gpd.GeoDataFrame],
    output_path: str,
    overwrite: bool = False
) -> Path:
    """
    Crea un solo GeoPackage con múltiples capas.
    
    Args:
        vectors_dict: {nombre_capa: GeoDataFrame}
        output_path: Ruta del GPKG de salida
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not overwrite:
        logger.warning(f"GPKG existe, omitiendo: {output_path}")
        return output_path
    
    for layer_name, gdf in vectors_dict.items():
        if len(gdf) > 0:
            # Sanitizar nombre de capa (GPKG no permite espacios, caracteres especiales)
            safe_name = layer_name.replace(' ', '_').replace('-', '_').lower()
            gdf.to_file(output_path, driver="GPKG", layer=safe_name)
            logger.info(f"Capa '{safe_name}' añadida a GPKG ({len(gdf)} features)")
    
    logger.info(f"GeoPackage consolidado creado: {output_path}")
    return output_path


def add_metadata_to_gpkg(
    gpkg_path: str,
    metadata: Dict
):
    """
    Añade metadatos a un GeoPackage existente (en tabla gpkg_metadata).
    """
    # GeoPackage estándar no tiene tabla metadata obligatoria
    # Guardamos como JSON sidecar
    meta_path = Path(gpkg_path).with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadatos guardados: {meta_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test rápido
    test_raster = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4]
    ], dtype=np.uint8)
    
    transform = Affine(0.0001, 0, -75, 0, -0.0001, 5)
    crs = rasterio.crs.CRS.from_epsg(4326)
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test transition raster
        gdf = transition_raster_to_polygons(
            test_raster * 10 + np.roll(test_raster, 1, axis=1),
            transform, crs
        )
        print(f"Polígonos generados: {len(gdf)}")
        print(gdf[['transition_code', 'transition_name', 'area_ha']])