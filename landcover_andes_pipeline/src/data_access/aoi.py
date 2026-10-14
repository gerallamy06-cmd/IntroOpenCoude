"""
Módulo AOI: Descarga y filtrado de shapefile región Andina desde GADM v4.1
"""
import logging
from pathlib import Path
from typing import Optional, List, Union
import geopandas as gpd
import pooch
import requests
from shapely.geometry import box
import zipfile
import io

logger = logging.getLogger(__name__)

# Departamentos andinos de Colombia (GADM nivel 1 - NAME_1)
ANDEAN_DEPARTMENTS = [
    "Antioquia", "Boyacá", "Caldas", "Cauca", "Cesar",
    "Cundinamarca", "Huila", "Nariño", "Norte de Santander",
    "Quindío", "Risaralda", "Santander", "Tolima", "Valle del Cauca"
]

GADM_CONFIG = {
    "version": "4.1",
    "country": "COL",
    "level": 1,
    "base_url": "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/",
    "filename": "gadm41_COL.gpkg"
}


def get_gadm_url(version: str = "4.1", country: str = "COL", level: int = 1) -> str:
    """Construye URL de descarga GADM."""
    if version == "4.1":
        return f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{country}.gpkg"
    elif version == "4.0":
        return f"https://geodata.ucdavis.edu/gadm/gadm4.0/gpkg/gadm40_{country}.gpkg"
    else:
        raise ValueError(f"Versión GADM no soportada: {version}")


def download_gadm(
    output_dir: Union[str, Path],
    version: str = "4.1",
    country: str = "COL",
    level: int = 1,
    progress: bool = True
) -> Path:
    """
    Descarga GADM GeoPackage usando pooch (con cache automático).
    
    Args:
        output_dir: Directorio donde guardar
        version: Versión GADM (4.1, 4.0)
        country: Código ISO país (COL)
        level: Nivel administrativo (0=país, 1=departamentos, 2=municipios)
        progress: Mostrar barra de progreso
    
    Returns:
        Path al archivo .gpkg descargado
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"gadm{version.replace('.', '')}_{country}.gpkg"
    filepath = output_dir / filename
    
    if filepath.exists():
        logger.info(f"GADM ya existe en cache: {filepath}")
        return filepath
    
    url = get_gadm_url(version, country, level)
    logger.info(f"Descargando GADM {version} para {country} desde {url}")
    
    try:
        # Usar pooch para descarga con cache y verificación
        GOOD_BOY = pooch.Pooch(
            path=output_dir,
            base_url="",
            registry={filename: None},  # Sin hash para flexibilidad
            urls={filename: url},
            downloader=pooch.HTTPDownloader(progressbar=progress)
        )
        GOOD_BOY.fetch(filename)
        logger.info(f"GADM descargado exitosamente: {filepath}")
        return filepath
    except Exception as e:
        logger.warning(f"Pooch falló, intentando descarga directa: {e}")
        return _download_direct(url, filepath, progress)


def _download_direct(url: str, filepath: Path, progress: bool = True) -> Path:
    """Descarga directa con requests como fallback."""
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        if progress and total_size > 0:
            from tqdm import tqdm
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="GADM") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    logger.info(f"Descarga directa completada: {filepath}")
    return filepath


def load_andean_aoi(
    gadm_path: Union[str, Path],
    departments: Optional[List[str]] = None,
    buffer_km: float = 0.0,
    simplify_tolerance: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Carga GADM y filtra departamentos andinos.
    
    Args:
        gadm_path: Ruta al .gpkg de GADM
        departments: Lista de nombres de departamentos (None = ANDEAN_DEPARTMENTS)
        buffer_km: Buffer en kilómetros alrededor del AOI
        simplify_tolerance: Tolerancia para simplificar geometrías (grados)
    
    Returns:
        GeoDataFrame con geometría unificada de la región andina
    """
    if departments is None:
        departments = ANDEAN_DEPARTMENTS
    
    logger.info(f"Cargando GADM desde: {gadm_path}")
    logger.info(f"Filtrando departamentos: {departments}")
    
    # Leer capa nivel 1 (departamentos)
    gdf = gpd.read_file(gadm_path, layer=f"ADM_{1}")
    
    # Verificar columna de nombre (puede ser NAME_1, ADM1_NAME, etc.)
    name_cols = [c for c in gdf.columns if 'name' in c.lower() or 'adm1' in c.lower()]
    name_col = name_cols[0] if name_cols else 'NAME_1'
    
    logger.debug(f"Columnas disponibles: {list(gdf.columns)}")
    logger.debug(f"Usando columna de nombre: {name_col}")
    
    # Filtrar departamentos andinos
    andean_gdf = gdf[gdf[name_col].isin(departments)].copy()
    
    if len(andean_gdf) != len(departments):
        found = set(andean_gdf[name_col].tolist())
        missing = set(departments) - found
        logger.warning(f"Departamentos no encontrados en GADM: {missing}")
        logger.info(f"Departamentos encontrados: {found}")
    
    # Unificar en una sola geometría
    aoi_union = andean_gdf.geometry.unary_union
    
    # Crear GeoDataFrame resultante
    result = gpd.GeoDataFrame(
        geometry=[aoi_union],
        crs=andean_gdf.crs
    )
    
    # Aplicar buffer si se solicita (en metros → convertir a grados aprox)
    if buffer_km > 0:
        # 1 grado ≈ 111 km en latitud
        buffer_deg = buffer_km / 111.0
        result.geometry = result.geometry.buffer(buffer_deg)
        logger.info(f"Buffer aplicado: {buffer_km} km ({buffer_deg:.4f}°)")
    
    # Simplificar si se solicita
    if simplify_tolerance > 0:
        result.geometry = result.geometry.simplify(simplify_tolerance, preserve_topology=True)
        logger.info(f"Geometría simplificada con tolerancia: {simplify_tolerance}")
    
    # Asegurar CRS WGS84
    if result.crs != "EPSG:4326":
        result = result.to_crs("EPSG:4326")
        logger.info("CRS reproyectado a EPSG:4326")
    
    logger.info(f"AOI Andina creado: {len(result)} feature(s), área: {result.geometry.area.iloc[0]:.4f}°²")
    return result


def get_aoi_bbox(aoi_gdf: gpd.GeoDataFrame) -> List[float]:
    """Obtiene bounding box [minx, miny, maxx, maxy] del AOI."""
    bounds = aoi_gdf.total_bounds
    return [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]


def save_aoi_shapefile(aoi_gdf: gpd.GeoDataFrame, output_path: Union[str, Path]) -> Path:
    """Guarda AOI como shapefile (para uso en QGIS/GEE)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Shapefile requiere archivos auxiliares
    aoi_gdf.to_file(output_path, driver="ESRI Shapefile", encoding="utf-8")
    logger.info(f"AOI guardado como shapefile: {output_path}")
    return output_path


def save_aoi_geopackage(aoi_gdf: gpd.GeoDataFrame, output_path: Union[str, Path], layer_name: str = "aoi_andina") -> Path:
    """Guarda AOI como GeoPackage."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    aoi_gdf.to_file(output_path, layer=layer_name, driver="GPKG")
    logger.info(f"AOI guardado como GeoPackage: {output_path} (capa: {layer_name})")
    return output_path


def create_aoi_from_config(config: dict, cache_dir: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    Función principal: crea AOI desde configuración.
    
    Args:
        config: Diccionario con configuración (sección 'aoi')
        cache_dir: Directorio para cache de GADM
    
    Returns:
        GeoDataFrame con AOI andina
    """
    aoi_config = config.get("aoi", {})
    
    # Si hay shapefile_path explícito, usarlo
    if aoi_config.get("shapefile_path"):
        path = Path(aoi_config["shapefile_path"])
        if path.exists():
            logger.info(f"Cargando AOI desde shapefile proporcionado: {path}")
            aoi = gpd.read_file(path)
            if aoi.crs != "EPSG:4326":
                aoi = aoi.to_crs("EPSG:4326")
            return aoi
        else:
            logger.warning(f"Shapefile no encontrado: {path}, usando GADM automático")
    
    # Auto-descargar GADM
    gadm_path = download_gadm(
        output_dir=cache_dir,
        version=aoi_config.get("gadm_version", "4.1"),
        country="COL",
        level=aoi_config.get("gadm_level", 1)
    )
    
    # Filtrar departamentos andinos
    aoi = load_andean_aoi(
        gadm_path=gadm_path,
        departments=aoi_config.get("andean_departments"),
        buffer_km=aoi_config.get("buffer_km", 0.0)
    )
    
    return aoi


if __name__ == "__main__":
    # Test rápido
    import tempfile
    logging.basicConfig(level=logging.INFO)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        aoi = create_aoi_from_config(
            {"aoi": {"gadm_version": "4.1", "buffer_km": 0}},
            cache_dir=tmpdir
        )
        print(f"AOI shape: {aoi.shape}")
        print(f"CRS: {aoi.crs}")
        print(f"Bounds: {aoi.total_bounds}")
        print(f"Departamentos: {len(ANDEAN_DEPARTMENTS)}")