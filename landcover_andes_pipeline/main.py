#!/usr/bin/env python3
"""
Pipeline Principal: Detección de Cobertura Terrestre Región Andina Colombia
Orquesta todo el flujo: AOI → Data Access → Processing → Export → Visualization
"""
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Imports locales
from src.data_access.aoi import create_aoi_from_config
from src.data_access.planetary_computer import get_lulc_stack
from src.processing.mosaic_clip import clip_stack_to_aoi, compute_valid_mask
from src.processing.remap_classes import remap_stack, validate_remap, print_class_summary
from src.processing.change_detection import (
    compute_annual_transitions, compute_global_transition,
    create_change_summary, save_transition_results
)
from src.processing.hotspots import run_hotspots_analysis, save_hotspots_results
from src.processing.stats import (
    get_class_statistics, calculate_transition_statistics,
    create_consolidated_report, save_statistics, print_summary
)
from src.visualization.interactive_map import create_interactive_map
from src.export.raster_export import (
    write_cog_from_xarray, write_transition_cog, write_gi_raster,
    validate_cog, batch_validate_cogs
)
from src.export.vector_export import (
    save_transition_vectors, save_hotspots_vectors,
    save_aoi_vector, export_transition_matrices_csv,
    export_statistics_csv, create_consolidated_geopackage
)

# Configurar logging
def setup_logging(log_file: str = "logs/pipeline.log", level: str = "INFO"):
    """Configura logging a archivo y consola."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reducir verbosidad de librerías externas
    logging.getLogger('rasterio').setLevel(logging.WARNING)
    logging.getLogger('pystac_client').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('fiona').setLevel(logging.WARNING)
    logging.getLogger('pyproj').setLevel(logging.WARNING)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Carga configuración desde YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_pipeline(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Ejecuta pipeline completo.
    
    Returns:
        Dict con paths de todos los outputs generados
    """
    start_time = datetime.now()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE COBERTURA TERRESTRE REGIÓN ANDINA")
    logger.info("=" * 60)
    
    # 1. Cargar configuración
    config = load_config(config_path)
    logger.info(f"Configuración cargada: {config_path}")
    
    output_base = Path(config['output']['base_dir'])
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Subdirectorios
    dirs = {
        'rasters': output_base / 'rasters',
        'vectors': output_base / 'vectors',
        'stats': output_base / 'stats',
        'maps': output_base / 'maps',
        'logs': output_base / 'logs'
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # Reconfigurar logging a directorio de salida
    setup_logging(str(dirs['logs'] / f"pipeline_{start_time.strftime('%Y%m%d_%H%M%S')}.log"))
    logger = logging.getLogger(__name__)
    
    results = {
        'config': config,
        'start_time': start_time.isoformat(),
        'output_dirs': {k: str(v) for k, v in dirs.items()},
        'rasters': {},
        'vectors': {},
        'stats': {},
        'maps': {}
    }
    
    try:
        # ============================================================
        # 2. AOI: Cregar/descargar región Andina
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("PASO 1: ÁREA DE INTERÉS (AOI)")
        logger.info("=" * 40)
        
        aoi_gdf = create_aoi_from_config(config, cache_dir=dirs['rasters'] / 'gadm_cache')
        results['aoi'] = {
            'bounds': list(aoi_gdf.total_bounds),
            'crs': str(aoi_gdf.crs),
            'area_deg2': float(aoi_gdf.geometry.area.iloc[0])
        }
        
        # Guardar AOI
        aoi_gpkg = save_aoi_vector(aoi_gdf, dirs['vectors'])
        results['vectors']['aoi'] = str(aoi_gpkg)
        
        # ============================================================
        # 3. DATA ACCESS: Descargar stack LULC de Planetary Computer
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("PASO 2: DESCARGA DATOS LULC (Planetary Computer)")
        logger.info("=" * 40)
        
        stack = get_lulc_stack(aoi_gdf, config)
        results['stack_metadata'] = {
            'shape': dict(zip(['time', 'band', 'y', 'x'], stack.shape)),
            'years': [str(t)[:4] for t in stack.time.values],
            'crs': str(stack.rio.crs) if hasattr(stack, 'rio') else None
        }
        
        # ============================================================
        # 4. PROCESSING: Mosaico + Recorte AOI
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("PASO 3: MOSAICO Y RECORTE AL AOI")
        logger.info("=" * 40)
        
        clipped_stack = clip_stack_to_aoi(stack, aoi_gdf)
        results['clipped_stack_shape'] = dict(zip(['time', 'band', 'y', 'x'], clipped_stack.shape))
        
        # Máscara válida
        valid_mask = compute_valid_mask(
            clipped_stack,
            nodata_values=config['classes'].get('nodata_values', [0, 9, 10])
        )
        
        # ============================================================
        # 5. REMAPEO: IO-LULC 9 → MapBiomas 7
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("PASO 4: REMAPEO DE CLASES (IO-LULC → MapBiomas)")
        logger.info("=" * 40)
        
        remapped_stack = remap_stack(clipped_stack, config['classes'])
        validation = validate_remap(remapped_stack)
        results['remap_validation'] = validation
        
        print_class_summary(remapped_stack)
        
        # ============================================================
        # 6. EXPORT RASTERS ANUALES
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("PASO 5: EXPORTACIÓN RASTERS ANUALES (COG)")
        logger.info("=" * 40)
        
        annual_rasters = write_cog_from_xarray(
            remapped_stack,
            str(dirs['rasters']),
            prefix="landcover",
            overwrite=config['output'].get('overwrite', False)
        )
        results['rasters']['annual'] = {k: str(v) for k, v in annual_rasters.items()}
        
        # También guardar latest (último año)
        latest_year = max(annual_rasters.keys(), key=lambda x: int(x.split('_')[1].split('.')[0]))
        latest_src = annual_rasters[latest_year]
        import shutil
        latest_dst = dirs['rasters'] / 'landcover_latest.tif'
        shutil.copy2(latest_src, latest_dst)
        results['rasters']['latest'] = str(latest_dst)
        
        # ============================================================
        # 7. CHANGE DETECTION: Matrices + Rasters transición
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("PASO 6: DETECCIÓN DE CAMBIO (Matrices + Rasters)")
        logger.info("=" * 40)
        
        years = [int(str(t)[:4]) for t in remapped_stack.time.values]
        
        # Transiciones anuales consecutivas
        annual_change = compute_annual_transitions(
            remapped_stack, years,
            nodata_value=0
        )
        
        # Transición global 2017→2022
        global_change = compute_global_transition(
            remapped_stack, years,
            year_from=years[0],
            year_to=years[-1],
            nodata_value=0
        )
        
        change_results = {
            'annual': annual_change,
            'global': global_change
        }
        
        # Exportar rasters de transición
        if hasattr(clipped_stack, 'rio'):
            template_transform = clipped_stack.rio.transform()
            template_crs = clipped_stack.rio.crs
        else:
            template_transform = None
            template_crs = None
        
        for pair_key, trans_raster in annual_change['transition_rasters'].items():
            y_from, y_to = pair_key.split('_')
            write_transition_cog(
                trans_raster,
                str(dirs['rasters'] / f"transition_{pair_key}.tif"),
                template_transform, template_crs,
                int(y_from), int(y_to),
                overwrite=config['output'].get('overwrite', False)
            )
            results['rasters'][f'transition_{pair_key}'] = str(dirs['rasters'] / f"transition_{pair_key}.tif")
        
        # Global
        write_transition_cog(
            global_change['transition_raster'],
            str(dirs['rasters'] / "transition_2017_2022.tif"),
            template_transform, template_crs,
            years[0], years[-1],
            overwrite=config['output'].get('overwrite', False)
        )
        results['rasters']['transition_global'] = str(dirs['rasters'] / "transition_2017_2022.tif")
        
        # ============================================================
        # 8. HOTSPOTS: Getis-Ord Gi*
        # ============================================================
        hotspots_results = None
        if config['hotspots'].get('enabled', True):
            logger.info("\n" + "=" * 40)
            logger.info("PASO 7: ANÁLISIS HOTSPOTS (Getis-Ord Gi*)")
            logger.info("=" * 40)
            
            hotspots_results = run_hotspots_analysis(
                global_change['transition_raster'],
                template_transform,
                str(template_crs),
                config['hotspots']
            )
            
            if 'error' not in hotspots_results:
                # Guardar raster Gi* interpolado
                if hotspots_results.get('gi_raster') is not None:
                    write_gi_raster(
                        hotspots_results['gi_raster'],
                        str(dirs['rasters'] / 'hotspots_gistar.tif'),
                        template_transform, template_crs,
                        config['hotspots'],
                        overwrite=config['output'].get('overwrite', False)
                    )
                    results['rasters']['hotspots_gistar'] = str(dirs['rasters'] / 'hotspots_gistar.tif')
                
                # Guardar vectores hotspots
                hs_vectors = save_hotspots_results(hotspots_results, str(dirs['vectors']))
                results['vectors'].update({f'hotspots_{k}': str(v) for k, v in hs_vectors.items()})
                logger.info(f"Hotspots guardados: {list(hs_vectors.keys())}")
            else:
                logger.warning(f"Hotspots falló: {hotspots_results['error']}")
        
        # ============================================================
        # 9. VECTORES: Transiciones
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("PASO 8: EXPORTACIÓN VECTORES (GPKG)")
        logger.info("=" * 40)
        
        trans_vectors = save_transition_vectors(
            change_results, str(dirs['vectors']), clipped_stack,
            overwrite=config['output'].get('overwrite', False)
        )
        results['vectors'].update({k: str(v) for k, v in trans_vectors.items()})
        
        # ============================================================
        # 10. ESTADÍSTICAS
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("PASO 9: CÁLCULO Y EXPORTACIÓN ESTADÍSTICAS")
        logger.info("=" * 40)
        
        # Área por clase/año
        pixel_area = 0.01  # 10m x 10m default
        if template_transform:
            from src.processing.stats import calculate_pixel_area_ha
            pixel_area = calculate_pixel_area_ha(template_transform, str(template_crs))
        
        area_stats = get_class_statistics(remapped_stack, pixel_area_ha=pixel_area)
        trans_stats = calculate_transition_statistics(change_results, pixel_area_ha=pixel_area)
        
        # Imprimir resumen
        print_summary(area_stats, trans_stats)
        
        # Exportar CSVs
        stat_files = export_statistics_csv(
            area_stats, trans_stats, hotspots_results, str(dirs['stats']),
            overwrite=config['output'].get('overwrite', False)
        )
        results['stats'].update({k: str(v) for k, v in stat_files.items()})
        
        # Matrices CSV
        matrix_files = export_transition_matrices_csv(change_results, str(dirs['stats']))
        results['stats'].update({k: str(v) for k, v in matrix_files.items()})
        
        # Reporte consolidado JSON
        report = create_consolidated_report(area_stats, trans_stats, hotspots_results, config)
        report_path = dirs['stats'] / 'consolidated_report.json'
        with open(report_path, 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        results['stats']['consolidated_report'] = str(report_path)
        
        # ============================================================
        # 11. MAPA INTERACTIVO
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("PASO 10: MAPA INTERACTIVO (HTML)")
        logger.info("=" * 40)
        
        map_path = dirs['maps'] / "andes_landcover_2017_2022_hotspots.html"
        
        annual_raster_paths = {int(k.split('_')[1].split('.')[0]): v for k, v in annual_rasters.items()}
        
        create_interactive_map(
            output_path=str(map_path),
            annual_rasters=annual_raster_paths,
            change_raster=str(dirs['rasters'] / "transition_2017_2022.tif"),
            hotspots_gpkg=results['vectors'].get('hotspots_hotspots_deforestation'),
            gi_raster=results['rasters'].get('hotspots_gistar'),
            aoi_gpkg=str(aoi_gpkg),
            config=config
        )
        results['maps']['interactive_html'] = str(map_path)
        
        # ============================================================
        # 12. VALIDACIÓN COGs
        # ============================================================
        logger.info("\n" + "=" * 40)
        logger.info("PASO 11: VALIDACIÓN COGs GENERADOS")
        logger.info("=" * 40)
        
        validation_results = batch_validate_cogs(str(dirs['rasters']))
        results['cog_validation'] = validation_results
        
        # ============================================================
        # FINALIZAR
        # ============================================================
        end_time = datetime.now()
        duration = end_time - start_time
        
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = duration.total_seconds()
        results['duration_formatted'] = str(duration)
        results['status'] = 'SUCCESS'
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info(f"Duración total: {duration}")
        logger.info(f"Outputs en: {output_base}")
        logger.info("=" * 60)
        
        # Guardar resumen de ejecución
        summary_path = output_base / 'pipeline_summary.json'
        with open(summary_path, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        
        return results
        
    except Exception as e:
        logger.exception(f"ERROR EN PIPELINE: {e}")
        results['status'] = 'ERROR'
        results['error'] = str(e)
        results['end_time'] = datetime.now().isoformat()
        raise


def main():
    """Entry point CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pipeline Cobertura Terrestre Región Andina Colombia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                    # Usa config.yaml por defecto
  python main.py -c mi_config.yaml  # Config personalizada
  python main.py --step aoi         # Solo paso AOI
  python main.py --step map         # Solo generar mapa (requiere outputs previos)
        """
    )
    
    parser.add_argument('-c', '--config', default='config.yaml',
                        help='Ruta al archivo de configuración YAML')
    parser.add_argument('--step', choices=['aoi', 'download', 'process', 'change', 'hotspots', 'export', 'map', 'all'],
                        default='all', help='Ejecutar solo un paso específico')
    parser.add_argument('-v', '--verbose', action='store_true', help='Logging DEBUG')
    parser.add_argument('--overwrite', action='store_true', help='Sobrescribir outputs existentes')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(level=log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Iniciando pipeline con config: {args.config}")
    logger.info(f"Paso: {args.step}")
    
    if args.step == 'all':
        run_pipeline(args.config)
    else:
        logger.warning(f"Paso individual '{args.step}' no implementado aún. Ejecutando pipeline completo.")
        run_pipeline(args.config)


if __name__ == "__main__":
    main()