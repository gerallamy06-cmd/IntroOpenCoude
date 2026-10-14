"""
Módulo Mosaico y Recorte: Une tiles stackstac y recorta al AOI exacto
"""
import logging
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import xarray as xr
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.enums import Compression
import geopandas as gpd
from shapely.geometry import mapping, box
import dask.array as da

logger = logging.getLogger(__name__)


def ensure_rio(stack: xr.DataArray) -> xr.DataArray:
    """Asegura que el stack tenga acceso rioxarray."""
    try:
        import rioxarray  # noqa: F401
    except ImportError:
        logger.warning("rioxarray no instalado, algunas funciones pueden fallar")
    return stack


def mosaic_tiles_lazy(stack: xr.DataArray) -> xr.DataArray:
    """
    Mosaico de tiles usando stackstac (ya maneja overlapping internamente).
    stackstac.stack() ya hace el mosaico automáticamente al combinar tiles.
    
    Esta función verifica y consolida si hay múltiples bandas/tiempos.
    """
    # stackstac ya retorna un mosaico unificado
    # Solo validamos que no haya NaNs excesivos
    logger.info(f"Stack recibido: {stack.shape}, chunks: {stack.chunks}")
    
    # Verificar cobertura de datos válidos
    valid_pct = (stack != 0).sum() / stack.size * 100
    logger.info(f"Porcentaje de píxeles válidos (≠0): {valid_pct.compute():.2f}%")
    
    return stack


def clip_stack_to_aoi(
    stack: xr.DataArray,
    aoi_gdf: gpd.GeoDataFrame,
    nodata: int = 0,
    all_touched: bool = False,
    crop: bool = True
) -> xr.DataArray:
    """
    Recorta el stack al AOI exacto usando rasterio.mask.
    
    Args:
        stack: xarray.DataArray (time, band, y, x) con rio accessor
        aoi_gdf: GeoDataFrame con geometría AOI (mismo CRS que stack)
        nodata: Valor nodata
        all_touched: Si True, incluye píxeles tocados por el borde
        crop: Si True, recorta la extensión al bounding box de la geometría
    
    Returns:
        xarray.DataArray recortado
    """
    logger.info("Recortando stack al AOI...")
    
    # Asegurar mismo CRS
    if hasattr(stack, 'rio'):
        stack_crs = stack.rio.crs
        if stack_crs != aoi_gdf.crs:
            logger.info(f"Reproyectando AOI de {aoi_gdf.crs} a {stack_crs}")
            aoi_gdf = aoi_gdf.to_crs(stack_crs)
    
    # Obtener geometrías como lista de dicts (formato rasterio)
    geometries = [mapping(geom) for geom in aoi_gdf.geometry]
    
    # Recortar usando rioxarray (usa rasterio.mask internamente)
    try:
        clipped = stack.rio.clip(
            geometries=geometries,
            crs=aoi_gdf.crs,
            drop=True,        # Recortar extensión
            all_touched=all_touched
        )
        logger.info(f"Stack recortado: {clipped.shape}")
        return clipped
    except Exception as e:
        logger.error(f"Error en rio.clip: {e}")
        # Fallback manual con rasterio
        return _clip_stack_manual(stack, geometries, nodata, crop)


def _clip_stack_manual(
    stack: xr.DataArray,
    geometries: List[Dict],
    nodata: int,
    crop: bool
) -> xr.DataArray:
    """Recorte manual usando rasterio.mask directo (fallback)."""
    logger.info("Usando recorte manual con rasterio.mask...")
    
    # Convertir a numpy array (compute si es dask)
    data = stack.data
    if hasattr(data, 'compute'):
        logger.info("Computando stack dask para recorte manual...")
        data = data.compute()
    
    # data shape: (time, band, y, x)
    n_time, n_band, height, width = data.shape
    
    # Obtener transform y CRS del stack
    if hasattr(stack, 'rio'):
        transform = stack.rio.transform()
        crs = stack.rio.crs
    else:
        # Fallback: calcular desde coords
        x_coords = stack.x.values
        y_coords = stack.y.values
        transform = from_bounds(
            x_coords.min(), y_coords.min(),
            x_coords.max(), y_coords.max(),
            width, height
        )
        crs = "EPSG:4326"
    
    # Recortar cada banda temporal
    clipped_bands = []
    for t in range(n_time):
        band_data = data[t, 0]  # asumiendo 1 banda
        out_image, out_transform = mask(
            rasterio.io.MemoryFile().open(
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=band_data.dtype,
                crs=crs,
                transform=transform,
                nodata=nodata
            ),
            geometries,
            crop=crop,
            nodata=nodata,
            all_touched=False
        )
        clipped_bands.append(out_image[0])  # (1, H, W) -> (H, W)
    
    # Reconstruir DataArray
    clipped_data = np.stack(clipped_bands, axis=0)  # (time, H, W)
    clipped_data = clipped_data[:, np.newaxis, :, :]  # (time, band, H, W)
    
    # Crear nuevas coordenadas
    new_height, new_width = clipped_data.shape[2], clipped_data.shape[3]
    x_res = transform[0]
    y_res = -transform[4]  # negativo porque y aumenta hacia abajo
    
    new_x = np.arange(new_width) * x_res + out_transform[2]
    new_y = np.arange(new_height) * (-y_res) + out_transform[5]
    
    clipped_da = xr.DataArray(
        clipped_data,
        dims=["time", "band", "y", "x"],
        coords={
            "time": stack.time.values,
            "band": stack.band.values,
            "y": new_y,
            "x": new_x
        }
    )
    
    # Restaurar atributos rio
    clipped_da.rio.write_crs(crs, inplace=True)
    clipped_da.rio.write_transform(out_transform, inplace=True)
    clipped_da.rio.write_nodata(nodata, inplace=True)
    
    logger.info(f"Recorte manual completado: {clipped_da.shape}")
    return clipped_da


def reproject_match(
    stack: xr.DataArray,
    target_crs: str = "EPSG:3116",  # Colombia CRS para cálculos de área
    resolution: Optional[float] = None,
    resampling: Resampling = Resampling.nearest
) -> xr.DataArray:
    """
    Reproyecta stack a CRS objetivo (ej. EPSG:3116 para Colombia).
    Útil para cálculos de área precisos en hectáreas.
    """
    logger.info(f"Reproyectando a {target_crs}...")
    
    if hasattr(stack, 'rio'):
        reprojected = stack.rio.reproject(
            target_crs,
            resolution=resolution,
            resampling=resampling,
            nodata=0
        )
        logger.info(f"Reproyectado: {reprojected.shape}")
        return reprojected
    else:
        logger.warning("Stack no tiene rio accessor, no se puede reproyectar")
        return stack


def compute_valid_mask(stack: xr.DataArray, nodata_values: List[int] = None) -> xr.DataArray:
    """
    Calcula máscara de píxeles válidos (excluye nodata, nieve, nubes).
    
    Args:
        stack: DataArray (time, band, y, x)
        nodata_values: Valores a considerar como inválidos (default: [0, 9, 10])
    
    Returns:
        DataArray booleano (time, y, x) - True = válido
    """
    if nodata_values is None:
        nodata_values = [0, 9, 10]  # NoData, Snow/Ice, Clouds
    
    # Asumir banda 0
    data = stack.isel(band=0)
    
    # Máscara válida: no está en nodata_values
    valid_mask = ~data.isin(nodata_values)
    
    logger.info(f"Máscara válida calculada. Píxeles válidos: {valid_mask.sum().compute().item():,}")
    return valid_mask


def get_stack_bounds(stack: xr.DataArray) -> Tuple[float, float, float, float]:
    """Obtiene bounds (minx, miny, maxx, maxy) del stack."""
    if hasattr(stack, 'rio'):
        return stack.rio.bounds()
    else:
        x_min, x_max = float(stack.x.min()), float(stack.x.max())
        y_min, y_max = float(stack.y.min()), float(stack.y.max())
        return (x_min, y_min, x_max, y_max)


def get_stack_transform(stack: xr.DataArray):
    """Obtiene transform afín del stack."""
    if hasattr(stack, 'rio'):
        return stack.rio.transform()
    return None


def stack_to_numpy(stack: xr.DataArray, band: int = 0) -> np.ndarray:
    """
    Convierte stack a numpy array (time, y, x) computando si es dask.
    """
    data = stack.isel(band=band).data
    if hasattr(data, 'compute'):
        logger.info("Computando array dask a numpy...")
        data = data.compute()
    return data  # (time, y, x)


def numpy_to_stack(
    data: np.ndarray,
    template_stack: xr.DataArray,
    band: int = 0
) -> xr.DataArray:
    """
    Convierte numpy array de vuelta a DataArray usando template para coords.
    data shape: (time, y, x) o (y, x)
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]  # (1, y, x)
    
    n_time = data.shape[0]
    time_coords = template_stack.time.values[:n_time] if n_time <= len(template_stack.time) else template_stack.time.values
    
    new_stack = xr.DataArray(
        data[:, np.newaxis, :, :],  # (time, 1, y, x)
        dims=["time", "band", "y", "x"],
        coords={
            "time": time_coords,
            "band": template_stack.band.values,
            "y": template_stack.y.values,
            "x": template_stack.x.values
        }
    )
    
    # Copiar atributos rio
    if hasattr(template_stack, 'rio'):
        new_stack.rio.write_crs(template_stack.rio.crs, inplace=True)
        new_stack.rio.write_transform(template_stack.rio.transform(), inplace=True)
        new_stack.rio.write_nodata(template_stack.rio.nodata, inplace=True)
    
    return new_stack


def create_mosaic_preview(
    stack: xr.DataArray,
    year_idx: int = -1,
    band: int = 0,
    cmap: str = "viridis"
) -> np.ndarray:
    """
    Crea preview RGB del mosaico para visualización rápida.
    Retorna array (H, W, 3) normalizado 0-1.
    """
    import matplotlib.pyplot as plt
    
    data = stack.isel(time=year_idx, band=band).data
    if hasattr(data, 'compute'):
        data = data.compute()
    
    # Normalizar para visualización
    vmin, vmax = np.nanpercentile(data[data > 0], [2, 98]) if (data > 0).any() else (0, 1)
    norm_data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    # Aplicar colormap
    cmap_obj = plt.get_cmap(cmap)
    rgb = cmap_obj(norm_data)[:, :, :3]  # (H, W, 3)
    
    return rgb


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Módulo mosaic_clip listo para importar")