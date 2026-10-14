"""
Módulo Planetary Computer: Búsqueda STAC, descarga tiles IO-LULC con retry/backoff
"""
import logging
import time
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import xarray as xr
import stackstac
import pystac_client
import planetary_computer as pc
from pystac.extensions.eo import EOExtension as eo
from shapely.geometry import mapping
import rasterio
from rasterio.warp import transform_bounds
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Configuración por defecto
DEFAULT_CONFIG = {
    "collection": "io-lulc-annual-v02",
    "years": [2017, 2018, 2019, 2020, 2021, 2022],
    "asset_name": "data",  # asset principal en IO-LULC
    "chunk_size": 256,
    "resolution": 10,  # metros
    "max_cloud_cover": 100,
    "timeout": 300,
    "max_retries": 5,
    "backoff_factor": 2,
    "api_key": None
}

# Mapeo de años a rangos datetime para STAC
YEAR_DATETIME_RANGES = {
    2017: "2017-01-01T00:00:00Z/2017-12-31T23:59:59Z",
    2018: "2018-01-01T00:00:00Z/2018-12-31T23:59:59Z",
    2019: "2019-01-01T00:00:00Z/2019-12-31T23:59:59Z",
    2020: "2020-01-01T00:00:00Z/2020-12-31T23:59:59Z",
    2021: "2021-01-01T00:00:00Z/2021-12-31T23:59:59Z",
    2022: "2022-01-01T00:00:00Z/2022-12-31T23:59:59Z",
}

# Clases IO-LULC Annual v02 (9 clases + nodata)
IO_LULC_CLASSES = {
    0: "No Data",
    1: "Water",
    2: "Trees",
    3: "Grass",
    4: "Flooded Vegetation",
    5: "Crops",
    6: "Scrub/Shrub",
    7: "Built Area",
    8: "Bare Ground",
    9: "Snow/Ice",
    10: "Clouds",
    11: "Rangeland"
}


@dataclass
class STACItemInfo:
    """Información de un item STAC relevante."""
    id: str
    year: int
    datetime: str
    bbox: List[float]
    geometry: Dict
    assets: Dict
    tile_id: str  # ej: "10S_498000_499000" o similar


def create_stac_client(api_key: Optional[str] = None, timeout: int = 300) -> pystac_client.Client:
    """Crea cliente STAC configurado para Planetary Computer."""
    catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    
    modifier = pc.sign_inplace if api_key is None else None
    
    client = pystac_client.Client.open(
        catalog_url,
        modifier=modifier,
        headers={"User-Agent": "landcover-andes-pipeline/1.0"}
    )
    
    # Configurar timeout
    client.session.timeout = timeout
    
    if api_key:
        os.environ["PC_SDK_SUBSCRIPTION_KEY"] = api_key
        logger.info("Usando API key de Planetary Computer")
    else:
        logger.info("Usando acceso público (rate limited)")
    
    return client


def search_lulc_items(
    client: pystac_client.Client,
    aoi_geometry: Dict,
    years: List[int],
    collection: str = "io-lulc-annual-v02",
    max_cloud_cover: int = 100
) -> Dict[int, List[STACItemInfo]]:
    """
    Busca items IO-LULC que intersectan el AOI para cada año.
    
    Returns:
        Dict: {año: [STACItemInfo, ...]}
    """
    results = {}
    
    for year in years:
        datetime_range = YEAR_DATETIME_RANGES.get(year)
        if not datetime_range:
            logger.warning(f"No hay rango datetime definido para año {year}")
            continue
        
        logger.info(f"Buscando items {collection} para año {year}...")
        
        try:
            search = client.search(
                collections=[collection],
                intersects=aoi_geometry,
                datetime=datetime_range,
                query={"eo:cloud_cover": {"lt": max_cloud_cover}} if max_cloud_cover < 100 else None
            )
            
            items = list(search.items())
            
            if not items:
                logger.warning(f"No se encontraron items para año {year}")
                results[year] = []
                continue
            
            # Convertir a STACItemInfo
            item_infos = []
            for item in items:
                # Extraer tile_id del ID o properties
                tile_id = item.properties.get("tile_id", item.id)
                
                info = STACItemInfo(
                    id=item.id,
                    year=year,
                    datetime=str(item.datetime) if item.datetime else "",
                    bbox=list(item.bbox) if item.bbox else [],
                    geometry=mapping(item.geometry) if item.geometry else {},
                    assets=item.assets,
                    tile_id=tile_id
                )
                item_infos.append(info)
            
            logger.info(f"Año {year}: {len(item_infos)} tiles encontrados")
            results[year] = item_infos
            
        except Exception as e:
            logger.error(f"Error buscando items año {year}: {e}")
            results[year] = []
    
    return results


def load_lulc_stack(
    items_by_year: Dict[int, List[STACItemInfo]],
    aoi_bounds: Tuple[float, float, float, float],
    asset_name: str = "data",
    chunk_size: int = 256,
    resolution: float = 10.0,
    epsg: int = 4326
) -> xr.DataArray:
    """
    Carga stack de datos LULC usando stackstac (lazy loading).
    
    Args:
        items_by_year: Items organizados por año
        aoi_bounds: (minx, miny, maxx, maxy) en EPSG:4326
        asset_name: Nombre del asset a cargar ("data" para IO-LULC)
        chunk_size: Tamaño de chunk para dask
        resolution: Resolución en grados (10m ≈ 0.00009° en Ecuador)
        epsg: CRS de salida
    
    Returns:
        xarray.DataArray con dims (time, band, y, x)
    """
    # Aplanar items manteniendo orden temporal
    all_items = []
    time_coords = []
    
    for year in sorted(items_by_year.keys()):
        for item in items_by_year[year]:
            all_items.append(item)
            time_coords.append(np.datetime64(f"{year}-01-01"))
    
    if not all_items:
        raise ValueError("No hay items para cargar")
    
    logger.info(f"Cargando stack con {len(all_items)} tiles usando stackstac...")
    
    # Convertir items a formato STAC dict para stackstac
    stac_dicts = []
    for item in all_items:
        # Reconstruir item STAC mínimo
        stac_dict = {
            "type": "Feature",
            "id": item.id,
            "geometry": item.geometry,
            "bbox": item.bbox,
            "properties": {
                "datetime": item.datetime
            },
            "assets": {}
        }
        # Solo incluir asset solicitado
        if asset_name in item.assets:
            stac_dict["assets"][asset_name] = item.assets[asset_name].to_dict()
        stac_dicts.append(stac_dict)
    
    # Calcular resolución en grados
    # 10m en Ecuador ≈ 10/111111 ≈ 0.00009°
    res_deg = resolution / 111111.0
    
    # Cargar con stackstac (lazy - retorna dask array)
    stack = stackstac.stack(
        stac_dicts,
        assets=[asset_name],
        bounds=aoi_bounds,
        resolution=res_deg,
        epsg=epsg,
        chunksize=chunk_size,
        dtype="uint8",
        fill_value=0
    )
    
    # Asignar coordenadas de tiempo
    stack = stack.assign_coords(time=("time", time_coords))
    stack = stack.rename({"band": "band", "y": "y", "x": "x"})
    
    logger.info(f"Stack cargado: {stack.shape} (time, band, y, x)")
    logger.info(f"Chunks: {stack.chunks}")
    
    return stack


def compute_stack_with_retry(
    items_by_year: Dict[int, List[STACItemInfo]],
    aoi_bounds: Tuple[float, float, float, float],
    config: Dict,
    progress: bool = True
) -> xr.DataArray:
    """
    Computa el stack con reintentos exponenciales para rate limiting.
    """
    max_retries = config.get("max_retries", 5)
    backoff_factor = config.get("backoff_factor", 2)
    base_delay = 2  # segundos
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            stack = load_lulc_stack(
                items_by_year=items_by_year,
                aoi_bounds=aoi_bounds,
                asset_name=config.get("asset_name", "data"),
                chunk_size=config.get("chunk_size", 256),
                resolution=config.get("resolution", 10.0)
            )
            
            # Trigger computation para verificar que funciona
            # Solo compute un pequeño sample
            sample = stack.isel(time=0, band=0, y=slice(0, 10), x=slice(0, 10))
            _ = sample.compute()
            
            logger.info("Stack verificado exitosamente")
            return stack
            
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            
            # Detectar rate limiting (429) o errores de servidor (5xx)
            is_rate_limit = "429" in error_msg or "rate limit" in error_msg
            is_server_error = any(code in error_msg for code in ["500", "502", "503", "504"])
            is_timeout = "timeout" in error_msg or "timed out" in error_msg
            
            if attempt < max_retries and (is_rate_limit or is_server_error or is_timeout):
                delay = base_delay * (backoff_factor ** attempt)
                # Añadir jitter aleatorio
                import random
                delay += random.uniform(0, 1)
                
                logger.warning(
                    f"Intento {attempt + 1}/{max_retries + 1} falló: {e}. "
                    f"Reintentando en {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Falló después de {attempt + 1} intentos: {e}")
                raise
    
    raise last_exception


def get_lulc_stack(
    aoi_gdf,  # GeoDataFrame
    config: Dict,
    cache_dir: Optional[Path] = None
) -> xr.DataArray:
    """
    Función principal: obtiene stack LULC completo para AOI y años configurados.
    
    Args:
        aoi_gdf: GeoDataFrame con geometría AOI (EPSG:4326)
        config: Diccionario configuración (sección 'data' + 'planetary_computer')
        cache_dir: Directorio opcional para cache de items STAC
    
    Returns:
        xarray.DataArray (time, band, y, x) con datos LULC
    """
    # Merge configs
    pc_config = {**DEFAULT_CONFIG, **config.get("data", {}), **config.get("planetary_computer", {})}
    
    years = pc_config.get("years", [2017, 2018, 2019, 2020, 2021, 2022])
    collection = pc_config.get("collection", "io-lulc-annual-v02")
    
    logger.info(f"Iniciando descarga LULC para años: {years}")
    logger.info(f"Colección: {collection}")
    
    # Preparar geometría AOI para STAC
    aoi_geometry = mapping(aoi_gdf.geometry.unary_union)
    aoi_bounds = tuple(aoi_gdf.total_bounds)
    
    # Crear cliente STAC
    client = create_stac_client(
        api_key=pc_config.get("api_key"),
        timeout=pc_config.get("timeout", 300)
    )
    
    # Buscar items por año
    items_by_year = search_lulc_items(
        client=client,
        aoi_geometry=aoi_geometry,
        years=years,
        collection=collection,
        max_cloud_cover=pc_config.get("max_cloud_cover", 100)
    )
    
    # Verificar que hay datos para todos los años
    years_with_data = [y for y, items in items_by_year.items() if items]
    years_missing = [y for y, items in items_by_year.items() if not items]
    
    if years_missing:
        logger.warning(f"Años sin datos: {years_missing}")
    
    if not years_with_data:
        raise ValueError("No se encontraron datos LULC para ningún año")
    
    logger.info(f"Años con datos: {years_with_data}")
    
    # Cargar stack con retry
    stack = compute_stack_with_retry(
        items_by_year={y: items_by_year[y] for y in years_with_data},
        aoi_bounds=aoi_bounds,
        config=pc_config
    )
    
    return stack


def get_stack_metadata(stack: xr.DataArray) -> Dict:
    """Extrae metadatos útiles del stack."""
    return {
        "shape": dict(zip(["time", "band", "y", "x"], stack.shape)),
        "chunks": stack.chunks,
        "dtype": str(stack.dtype),
        "crs": str(stack.rio.crs) if hasattr(stack, 'rio') else "EPSG:4326",
        "transform": list(stack.rio.transform()) if hasattr(stack, 'rio') else None,
        "bounds": list(stack.rio.bounds()) if hasattr(stack, 'rio') else None,
        "time_coords": [str(t) for t in stack.coords["time"].values],
        "nodata": 0
    }


if __name__ == "__main__":
    # Test rápido (requiere conexión)
    logging.basicConfig(level=logging.INFO)
    
    # Verificar conexión básica
    try:
        client = create_stac_client()
        collections = list(client.get_collections())
        lulc_colls = [c.id for c in collections if "lulc" in c.id.lower() or "landcover" in c.id.lower()]
        print(f"Colecciones LULC disponibles: {lulc_colls}")
    except Exception as e:
        print(f"Error de conexión: {e}")