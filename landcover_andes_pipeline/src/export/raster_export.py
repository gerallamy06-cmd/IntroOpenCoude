"""
Módulo Exportación Raster: Guarda GeoTIFFs COG con metadatos completos
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import xarray as xr
import rasterio
from rasterio.enums import Compression, Predictor
from rasterio.crs import CRS
from rasterio.transform import Affine
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Configuración COG por defecto
COG_PROFILE = {
    'driver': 'COG',
    'compress': 'DEFLATE',
    'predictor': Predictor.YES,  # Predictor 2 para datos enteros
    'blocksize': 256,
    'overview_resampling': 'nearest',
    'overview_blocksize': 256,
    'num_threads': 'ALL_CPUS',
    'bigtiff': 'IF_NEEDED'
}

# Metadatos estándar
STANDARD_TAGS = {
    'AREA_OR_POINT': 'Area',
    'CREATION_DATE': datetime.now().isoformat(),
    'SOFTWARE': 'landcover_andes_pipeline v1.0',
    'SOURCE': 'Planetary Computer - Esri 10m IO-LULC Annual v02',
    'PROCESSING': 'Remapeado a MapBiomas Colombia (7 clases)'
}


def prepare_cog_profile(
    dtype: str = 'uint8',
    nodata: int = 0,
    crs: CRS = None,
    transform: Affine = None,
    width: int = None,
    height: int = None,
    count: int = 1,
    extra_tags: Dict = None
) -> Dict:
    """
    Prepara perfil completo para escritura COG.
    """
    profile = {
        'driver': 'COG',
        'dtype': dtype,
        'nodata': nodata,
        'width': width,
        'height': height,
        'count': count,
        'crs': crs,
        'transform': transform,
        **COG_PROFILE
    }
    
    # Añadir tags
    tags = {**STANDARD_TAGS}
    if extra_tags:
        tags.update(extra_tags)
    profile['tags'] = tags
    
    return profile


def write_cog(
    array: np.ndarray,
    output_path: str,
    transform: Affine,
    crs: CRS,
    nodata: int = 0,
    dtype: str = 'uint8',
    tags: Dict = None,
    overwrite: bool = False
) -> Path:
    """
    Escribe array numpy como Cloud Optimized GeoTIFF.
    
    Args:
        array: Array 2D (H, W) o 3D (bands, H, W)
        output_path: Ruta de salida
        transform: Transform afín
        crs: CRS (rasterio.crs.CRS)
        nodata: Valor NoData
        dtype: Tipo de dato
        tags: Diccionario de metadatos adicionales
        overwrite: Sobrescribir si existe
    
    Returns:
        Path al archivo guardado
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not overwrite:
        logger.warning(f"Archivo existe, omitiendo (overwrite=False): {output_path}")
        return output_path
    
    # Asegurar dimensiones correctas
    if array.ndim == 2:
        array = array[np.newaxis, :, :]  # (1, H, W)
    elif array.ndim != 3:
        raise ValueError(f"Array debe ser 2D o 3D, recibido: {array.ndim}D")
    
    count, height, width = array.shape
    
    profile = prepare_cog_profile(
        dtype=dtype,
        nodata=nodata,
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        count=count,
        extra_tags=tags
    )
    
    logger.info(f"Escribiendo COG: {output_path} ({width}x{height}, {count} bandas, {dtype})")
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array.astype(dtype))
        
        # Escribir descripciones de bandas si hay múltiples
        if count > 1:
            for i in range(count):
                dst.set_band_description(i + 1, f"Band_{i+1}")
    
    logger.info(f"COG guardado exitosamente: {output_path}")
    return output_path


def write_cog_from_xarray(
    stack: xr.DataArray,
    output_dir: str,
    time_dim: str = 'time',
    band_dim: str = 'band',
    nodata: int = 0,
    overwrite: bool = False,
    prefix: str = "landcover"
) -> Dict[str, Path]:
    """
    Escribe cada time step de un DataArray xarray como COG individual.
    
    Args:
        stack: DataArray (time, band, y, x)
        output_dir: Directorio de salida
        time_dim: Nombre dimensión temporal
        band_dim: Nombre dimensión banda
        nodata: Valor NoData
        overwrite: Sobrescribir existentes
        prefix: Prefijo para nombres de archivo
    
    Returns:
        Dict {nombre_archivo: Path}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener coordenadas
    if hasattr(stack, 'rio'):
        transform = stack.rio.transform()
        crs = stack.rio.crs
    else:
        raise ValueError("Stack debe tener acceso rio (rioxarray)")
    
    times = stack[time_dim].values
    saved = {}
    
    for t_idx, time_val in enumerate(times):
        year = str(time_val)[:4]
        filename = f"{prefix}_{year}.tif"
        filepath = output_dir / filename
        
        # Extraer frame (band, y, x)
        frame = stack.isel({time_dim: t_idx})
        
        # Convertir a numpy
        data = frame.data
        if hasattr(data, 'compute'):
            data = data.compute()
        
        # data shape: (band, y, x)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]  # (y, x) - quitar banda si es 1
        
        # Tags específicos por año
        year_tags = {
            'YEAR': year,
            'TIME_INDEX': str(t_idx),
            'SOURCE_YEAR': year
        }
        
        write_cog(
            array=data,
            output_path=str(filepath),
            transform=transform,
            crs=crs,
            nodata=nodata,
            tags=year_tags,
            overwrite=overwrite
        )
        
        saved[filename] = filepath
    
    logger.info(f"Guardados {len(saved)} COGs anuales en {output_dir}")
    return saved


def write_transition_cog(
    transition_raster: np.ndarray,
    output_path: str,
    transform: Affine,
    crs: CRS,
    year_from: int,
    year_to: int,
    overwrite: bool = False
) -> Path:
    """
    Escribe raster de transiciones (códigos from*10+to) como COG.
    """
    tags = {
        'TRANSITION_TYPE': 'from10+to',
        'YEAR_FROM': str(year_from),
        'YEAR_TO': str(year_to),
        'CLASS_SCHEME': 'MapBiomas_Colombia_7clases',
        'CODE_DESCRIPTION': 'from_class * 10 + to_class (ej: 12 = Bosque(1) -> Pastizales(2))'
    }
    
    return write_cog(
        array=transition_raster,
        output_path=output_path,
        transform=transform,
        crs=crs,
        nodata=0,
        dtype='uint8',
        tags=tags,
        overwrite=overwrite
    )


def write_gi_raster(
    gi_raster: np.ndarray,
    output_path: str,
    transform: Affine,
    crs: CRS,
    config: Dict,
    overwrite: bool = False
) -> Path:
    """
    Escribe raster interpolado Gi* (valores continuos Z-score) como COG.
    Usa float32 para precisión.
    """
    tags = {
        'ANALYSIS_TYPE': 'Getis_Ord_GiStar',
        'WEIGHT_TYPE': config.get('weight_type', 'queen'),
        'DISTANCE_KM': str(config.get('distance_km', 1.0)),
        'PERMUTATIONS': str(config.get('permutations', 999)),
        'ALPHA': str(config.get('alpha', 0.05)),
        'TARGET_TRANSITIONS': str(config.get('target_transitions', [])),
        'VALUE_DESCRIPTION': 'Z-score Gi* (negativo=cold spot, positivo=hot spot)'
    }
    
    return write_cog(
        array=gi_raster,
        output_path=output_path,
        transform=transform,
        crs=crs,
        nodata=np.nan,
        dtype='float32',
        tags=tags,
        overwrite=overwrite
    )


def write_multiband_cog(
    arrays: Dict[str, np.ndarray],
    output_path: str,
    transform: Affine,
    crs: CRS,
    nodata: int = 0,
    overwrite: bool = False
) -> Path:
    """
    Escribe múltiples arrays como bandas de un solo COG multibanda.
    arrays: {nombre_banda: array_2d}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not overwrite:
        logger.warning(f"Archivo existe, omitiendo: {output_path}")
        return output_path
    
    # Verificar que todos tengan misma forma
    shapes = [arr.shape for arr in arrays.values()]
    if len(set(shapes)) > 1:
        raise ValueError(f"Todos los arrays deben tener misma forma: {shapes}")
    
    height, width = shapes[0]
    band_names = list(arrays.keys())
    count = len(band_names)
    
    # Stack arrays
    stacked = np.stack([arrays[name] for name in band_names], axis=0)  # (bands, H, W)
    
    tags = {
        'BAND_NAMES': ','.join(band_names),
        'MULTIBAND': 'true'
    }
    for i, name in enumerate(band_names):
        tags[f'BAND_{i+1}_NAME'] = name
    
    return write_cog(
        array=stacked,
        output_path=str(output_path),
        transform=transform,
        crs=crs,
        nodata=nodata,
        dtype='uint8',
        tags=tags,
        overwrite=overwrite
    )


def create_vrt_mosaic(
    input_files: List[str],
    output_vrt: str,
    overwrite: bool = False
) -> Path:
    """
    Crea VRT (Virtual Raster) mosaico de múltiples GeoTIFFs.
    Útil para visualización rápida sin copiar datos.
    """
    output_vrt = Path(output_vrt)
    output_vrt.parent.mkdir(parents=True, exist_ok=True)
    
    if output_vrt.exists() and not overwrite:
        logger.warning(f"VRT existe, omitiendo: {output_vrt}")
        return output_vrt
    
    import subprocess
    
    cmd = ['gdalbuildvrt', '-overwrite', str(output_vrt)] + input_files
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"gdalbuildvrt falló: {result.stderr}")
            raise RuntimeError(f"gdalbuildvrt error: {result.stderr}")
        logger.info(f"VRT mosaico creado: {output_vrt}")
    except FileNotFoundError:
        logger.error("gdalbuildvrt no encontrado. Instalar GDAL.")
        raise
    except subprocess.TimeoutExpired:
        logger.error("Timeout creando VRT")
        raise
    
    return output_vrt


def validate_cog(filepath: str) -> Dict:
    """
    Valida un COG y retorna metadatos.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return {'valid': False, 'error': 'File not found'}
    
    try:
        with rasterio.open(filepath) as src:
            info = {
                'valid': True,
                'path': str(filepath),
                'driver': src.driver,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtypes[0],
                'crs': str(src.crs) if src.crs else None,
                'transform': list(src.transform) if src.transform else None,
                'bounds': list(src.bounds) if src.bounds else None,
                'nodata': src.nodata,
                'compress': src.compression,
                'block_shape': src.block_shapes[0] if src.block_shapes else None,
                'overviews': src.overviews(1) if src.count > 0 else [],
                'tags': dict(src.tags()),
                'band_descriptions': src.descriptions if src.descriptions else []
            }
            
            # Verificar si es COG válido
            info['is_cog'] = src.driver == 'COG' or (
                src.is_tiled and 
                len(src.overviews(1)) > 0 if src.count > 0 else False
            )
            
            return info
            
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def batch_validate_cogs(directory: str, pattern: str = "*.tif") -> List[Dict]:
    """Valida todos los COGs en un directorio."""
    directory = Path(directory)
    results = []
    
    for filepath in directory.glob(pattern):
        result = validate_cog(str(filepath))
        result['filename'] = filepath.name
        results.append(result)
        
        if result['valid']:
            logger.info(f"  ✓ {filepath.name}: {result['width']}x{result['height']}, {result['count']}b, {result['dtype']}")
        else:
            logger.error(f"  ✗ {filepath.name}: {result.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test rápido
    test_array = np.random.randint(0, 8, (100, 100), dtype=np.uint8)
    test_transform = Affine(0.0001, 0, -75, 0, -0.0001, 5)
    test_crs = CRS.from_epsg(4326)
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_cog(
            test_array, 
            f"{tmpdir}/test.tif", 
            test_transform, 
            test_crs
        )
        print(f"Guardado: {path}")
        
        val = validate_cog(str(path))
        print(f"Validación: {val}")