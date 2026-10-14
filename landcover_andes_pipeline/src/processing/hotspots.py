"""
Módulo Hotspots: Análisis Getis-Ord Gi* para detección de clusters de cambio
Requiere: libpysal, esda, scipy
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import rasterio
from rasterio.features import shapes
from scipy import sparse

logger = logging.getLogger(__name__)

# Verificar dependencias
try:
    import libpysal
    from libpysal.weights import Queen, DistanceBand, KNN
    from esda.getisord import G_Local
    HAS_PYSAL = True
except ImportError:
    HAS_PYSAL = False
    logger.warning("libpysal/esda no disponible. Hotspots no funcionará.")
    logger.warning("Instalar con: conda install -c conda-forge libpysal esda")

# Transiciones de interés para hotspots (deforestación/urbanización)
TARGET_TRANSITIONS = [
    (1, 2),  # Bosque → Pastizales
    (1, 3),  # Bosque → Cultivos
    (1, 4),  # Bosque → Urbano
    (2, 4),  # Pastizales → Urbano
    (3, 4),  # Cultivos → Urbano
]


def check_dependencies():
    """Verifica dependencias necesarias."""
    if not HAS_PYSAL:
        raise ImportError(
            "libpysal y esda son requeridos para análisis de hotspots. "
            "Instalar: conda install -c conda-forge libpysal esda"
        )


def extract_change_points(
    transition_raster: np.ndarray,
    transform,
    crs: str,
    target_transitions: List[Tuple[int, int]] = None,
    sample_factor: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrae puntos de cambio del raster de transiciones.
    
    Args:
        transition_raster: Array (H, W) con códigos from*10+to
        transform: Transform afín rasterio
        crs: CRS del raster
        target_transitions: Lista de tuplas (from, to) a analizar
        sample_factor: Factor de submuestreo (1=todos, 2=1 de cada 2, etc.)
    
    Returns:
        (coords_x, coords_y, values) - coordenadas y valor binario (1=cambio objetivo)
    """
    if target_transitions is None:
        target_transitions = TARGET_TRANSITIONS
    
    # Crear códigos de transición objetivo
    target_codes = [f * 10 + t for f, t in target_transitions]
    
    # Máscara de cambios objetivo
    h, w = transition_raster.shape
    
    # Submuestreo sistemático para reducir puntos
    if sample_factor > 1:
        rows = np.arange(0, h, sample_factor)
        cols = np.arange(0, w, sample_factor)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        raster_sample = transition_raster[rr, cc]
    else:
        rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        raster_sample = transition_raster
    
    # Identificar píxeles con transiciones objetivo
    change_mask = np.isin(raster_sample, target_codes)
    
    # Convertir a binario: 1 = cambio objetivo (deforestación), 0 = otro/no cambio
    binary_values = change_mask.astype(np.uint8)
    
    # Obtener coordenadas geográficas solo de píxeles con datos
    valid_mask = (raster_sample != 0)  # Excluir NoData
    final_mask = change_mask & valid_mask
    
    # Coordenadas de centroides
    xs, ys = rasterio.transform.xy(transform, rr[final_mask], cc[final_mask])
    
    coords_x = np.array(xs)
    coords_y = np.array(ys)
    values = binary_values[final_mask]
    
    logger.info(f"Puntos de cambio extraídos: {len(values):,} "
                f"(positivos: {values.sum():,}, negativos: {(values==0).sum():,})")
    
    return coords_x, coords_y, values


def create_spatial_weights(
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    weight_type: str = "queen",
    distance_km: float = 1.0,
    k_neighbors: int = 8
) -> Any:
    """
    Crea matriz de pesos espaciales (W) para Getis-Ord Gi*.
    
    Args:
        coords_x, coords_y: Coordenadas de puntos
        weight_type: "queen", "rook", "distance", "knn"
        distance_km: Distancia en km para DistanceBand
        k_neighbors: k para KNN
    
    Returns:
        Objeto weights de libpysal
    """
    check_dependencies()
    
    # Crear array de coordenadas (n, 2)
    coords = np.column_stack([coords_x, coords_y])
    
    # Convertir distancia km a grados (aprox: 1° ≈ 111 km en Ecuador)
    distance_deg = distance_km / 111.0
    
    if weight_type.lower() == "queen":
        # Queen requiere polígonos, usamos DistanceBand como proxy
        # Para puntos, Queen ≈ DistanceBand con distancia diagonal
        w = DistanceBand(coords, threshold=distance_deg * np.sqrt(2), binary=True)
        logger.info(f"Pesos DistanceBand (Queen proxy): threshold={distance_deg*np.sqrt(2):.6f}°")
    
    elif weight_type.lower() == "rook":
        w = DistanceBand(coords, threshold=distance_deg, binary=True)
        logger.info(f"Pesos DistanceBand (Rook proxy): threshold={distance_deg:.6f}°")
    
    elif weight_type.lower() == "distance":
        w = DistanceBand(coords, threshold=distance_deg, binary=True)
        logger.info(f"Pesos DistanceBand: threshold={distance_deg:.6f}° ({distance_km} km)")
    
    elif weight_type.lower() == "knn":
        w = KNN(coords, k=k_neighbors)
        logger.info(f"Pesos KNN: k={k_neighbors}")
    
    else:
        # Default: DistanceBand 1km
        w = DistanceBand(coords, threshold=distance_deg, binary=True)
        logger.info(f"Pesos default DistanceBand: {distance_km} km")
    
    # Row-standardize (w_ij = 1/n_neighbors)
    w.transform = 'r'
    
    logger.info(f"Matriz pesos creada: {w.n} observaciones, "
                f"promedio vecinos: {w.mean_neighbors:.1f}")
    
    return w


def compute_getis_ord_gi(
    values: np.ndarray,
    weights,
    permutations: int = 999,
    n_jobs: int = 1
) -> Dict:
    """
    Calcula Getis-Ord Gi* local.
    
    Args:
        values: Array binario (0/1) de cambio objetivo
        weights: Matriz pesos libpysal
        permutations: Número de permutaciones para p-values simulados
        n_jobs: Paralelización (1 = secuencial)
    
    Returns:
        Dict con Gi*, Z-scores, p-values, clasificación
    """
    check_dependencies()
    
    logger.info(f"Calculando Getis-Ord Gi* con {permutations} permutaciones...")
    
    # Gi* local
    gi = G_Local(values, weights, transform='r', permutations=permutations, n_jobs=n_jobs)
    
    # Extraer resultados
    Gi = gi.Gs              # Gi* values
    Z = gi.Zs               # Z-scores
    p_sim = gi.p_sim        # p-values simulados (dos colas)
    
    # Clasificación de significancia (alpha=0.05, dos colas)
    # Hot spot: Z > 0 y p < 0.05
    # Cold spot: Z < 0 y p < 0.05
    # Not significant: p >= 0.05
    alpha = 0.05
    
    significance = np.zeros_like(Z, dtype=int)  # 0=NS, 1=Hot, -1=Cold
    significance[(Z > 0) & (p_sim < alpha)] = 1   # Hot spot
    significance[(Z < 0) & (p_sim < alpha)] = -1  # Cold spot
    
    # También clasificar por niveles de significancia
    sig_level = np.zeros_like(Z, dtype=int)
    sig_level[(Z > 0) & (p_sim < 0.001)] = 3    # Hot 99.9%
    sig_level[(Z > 0) & (p_sim < 0.01)] = 2     # Hot 99%
    sig_level[(Z > 0) & (p_sim < 0.05)] = 1     # Hot 95%
    sig_level[(Z < 0) & (p_sim < 0.001)] = -3   # Cold 99.9%
    sig_level[(Z < 0) & (p_sim < 0.01)] = -2    # Cold 99%
    sig_level[(Z < 0) & (p_sim < 0.05)] = -1    # Cold 95%
    
    results = {
        "Gi": Gi,
        "Z": Z,
        "p_sim": p_sim,
        "significance": significance,  # -1, 0, 1
        "sig_level": sig_level,        # -3,-2,-1,0,1,2,3
        "n_obs": len(values),
        "n_hot": int((significance == 1).sum()),
        "n_cold": int((significance == -1).sum()),
        "n_ns": int((significance == 0).sum()),
        "permutations": permutations,
        "alpha": alpha
    }
    
    logger.info(f"Gi* completado: Hot spots={results['n_hot']}, "
                f"Cold spots={results['n_cold']}, NS={results['n_ns']}")
    
    return results


def create_hotspots_geodataframe(
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    gi_results: Dict,
    crs: str,
    transition_raster: np.ndarray = None,
    transform = None
) -> gpd.GeoDataFrame:
    """
    Crea GeoDataFrame con resultados de hotspots.
    """
    # Geometrías puntos
    geometry = [Point(x, y) for x, y in zip(coords_x, coords_y)]
    
    # DataFrame base
    df = pd.DataFrame({
        "x": coords_x,
        "y": coords_y,
        "Gi": gi_results["Gi"],
        "Z_score": gi_results["Z"],
        "p_value": gi_results["p_sim"],
        "significance": gi_results["significance"],
        "sig_level": gi_results["sig_level"]
    })
    
    # Mapear significancia a etiquetas
    sig_labels = {
        -3: "Cold Spot 99.9%",
        -2: "Cold Spot 99%",
        -1: "Cold Spot 95%",
        0: "Not Significant",
        1: "Hot Spot 95%",
        2: "Hot Spot 99%",
        3: "Hot Spot 99.9%"
    }
    df["significance_label"] = df["sig_level"].map(sig_labels)
    
    # Si hay raster de transición, añadir código de transición
    if transition_raster is not None and transform is not None:
        # Obtener valor de transición para cada punto
        rows, cols = rasterio.transform.rowcol(transform, coords_x, coords_y)
        rows = np.clip(rows, 0, transition_raster.shape[0] - 1)
        cols = np.clip(cols, 0, transition_raster.shape[1] - 1)
        trans_codes = transition_raster[rows, cols]
        
        df["transition_code"] = trans_codes
        df["from_class"] = trans_codes // 10
        df["to_class"] = trans_codes % 10
        
        # Nombres de transición
        trans_names = {
            12: "Bosque→Pastizales", 13: "Bosque→Cultivos", 14: "Bosque→Urbano",
            24: "Pastizales→Urbano", 34: "Cultivos→Urbano",
            21: "Pastizales→Bosque", 31: "Cultivos→Bosque",
            72: "Humedales→Pastizales", 52: "Agua→Pastizales"
        }
        df["transition_name"] = df["transition_code"].map(trans_names)
    
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    
    logger.info(f"GeoDataFrame hotspots creado: {len(gdf)} features")
    return gdf


def interpolate_gi_to_raster(
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    gi_values: np.ndarray,
    transform,
    shape: Tuple[int, int],
    method: str = "linear"
) -> np.ndarray:
    """
    Interpola valores Gi* a raster completo usando scipy.interpolate.
    """
    try:
        from scipy.interpolate import griddata
    except ImportError:
        logger.warning("scipy.interpolate no disponible, no se puede interpolar")
        return None
    
    logger.info(f"Interpolando Gi* a raster {shape}...")
    
    # Crear grid de coordenadas del raster
    h, w = shape
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    grid_x, grid_y = rasterio.transform.xy(transform, rows, cols)
    grid_x = np.array(grid_x)
    grid_y = np.array(grid_y)
    
    # Interpolar
    points = np.column_stack([coords_x, coords_y])
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    interpolated = griddata(points, gi_values, grid_points, method=method, fill_value=0)
    interpolated = interpolated.reshape(shape)
    
    return interpolated


def run_hotspots_analysis(
    transition_raster: np.ndarray,
    transform,
    crs: str,
    config: Dict
) -> Dict[str, Any]:
    """
    Ejecuta análisis completo de hotspots Getis-Ord Gi*.
    
    Args:
        transition_raster: Raster códigos transición (H, W)
        transform: Transform afín
        crs: CRS
        config: Configuración hotspots (target_transitions, weight_type, etc.)
    
    Returns:
        Dict con resultados completos
    """
    check_dependencies()
    
    target_transitions = config.get("target_transitions", TARGET_TRANSITIONS)
    weight_type = config.get("weight_type", "queen")
    distance_km = config.get("distance_km", 1.0)
    permutations = config.get("permutations", 999)
    alpha = config.get("alpha", 0.05)
    sample_factor = config.get("sample_factor", 1)
    
    logger.info("=== INICIANDO ANÁLISIS HOTSPOTS Getis-Ord Gi* ===")
    logger.info(f"Transiciones objetivo: {target_transitions}")
    logger.info(f"Pesos: {weight_type}, distancia: {distance_km} km")
    logger.info(f"Permutaciones: {permutations}, alpha: {alpha}")
    logger.info(f"Factor submuestreo: {sample_factor}")
    
    # 1. Extraer puntos de cambio
    coords_x, coords_y, binary_values = extract_change_points(
        transition_raster=transition_raster,
        transform=transform,
        crs=crs,
        target_transitions=target_transitions,
        sample_factor=sample_factor
    )
    
    if len(binary_values) == 0:
        logger.warning("No se encontraron puntos de cambio objetivo")
        return {"error": "No change points found"}
    
    # 2. Crear pesos espaciales
    weights = create_spatial_weights(
        coords_x, coords_y,
        weight_type=weight_type,
        distance_km=distance_km
    )
    
    # 3. Calcular Getis-Ord Gi*
    gi_results = compute_getis_ord_gi(
        values=binary_values,
        weights=weights,
        permutations=permutations
    )
    
    # 4. Crear GeoDataFrame
    hotspots_gdf = create_hotspots_geodataframe(
        coords_x, coords_y, gi_results, crs,
        transition_raster=transition_raster,
        transform=transform
    )
    
    # 5. Interpolar Gi* a raster (opcional, para visualización)
    gi_raster = None
    if config.get("interpolate_raster", True) and len(coords_x) < 50000:
        gi_raster = interpolate_gi_to_raster(
            coords_x, coords_y, gi_results["Z"],
            transform, transition_raster.shape
        )
    
    # 6. Resumen estadístico
    summary = {
        "total_points_analyzed": len(binary_values),
        "positive_cases": int(binary_values.sum()),
        "negative_cases": int((binary_values == 0).sum()),
        "hot_spots_count": gi_results["n_hot"],
        "cold_spots_count": gi_results["n_cold"],
        "not_significant_count": gi_results["n_ns"],
        "hot_spots_pct": round(gi_results["n_hot"] / len(binary_values) * 100, 2),
        "cold_spots_pct": round(gi_results["n_cold"] / len(binary_values) * 100, 2),
        "mean_Gi": float(np.mean(gi_results["Gi"])),
        "mean_Z": float(np.mean(gi_results["Z"])),
        "config": config
    }
    
    # 7. Desglose por tipo de transición
    if "transition_code" in hotspots_gdf.columns:
        trans_summary = hotspots_gdf.groupby("transition_name").agg(
            n_points=("Gi", "count"),
            n_hot=("significance", lambda x: (x == 1).sum()),
            n_cold=("significance", lambda x: (x == -1).sum()),
            mean_Z=("Z_score", "mean")
        ).reset_index()
        summary["by_transition"] = trans_summary.to_dict("records")
    
    logger.info("=== ANÁLISIS HOTSPOTS COMPLETADO ===")
    logger.info(f"Hot spots: {summary['hot_spots_count']} ({summary['hot_spots_pct']}%)")
    logger.info(f"Cold spots: {summary['cold_spots_count']} ({summary['cold_spots_pct']}%)")
    
    return {
        "gi_results": gi_results,
        "hotspots_gdf": hotspots_gdf,
        "gi_raster": gi_raster,
        "summary": summary,
        "coords": (coords_x, coords_y),
        "values": binary_values
    }


def save_hotspots_results(
    hotspots_results: Dict,
    output_dir: str
) -> Dict[str, str]:
    """Guarda resultados de hotspots a disco."""
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved = {}
    
    # GeoDataFrame → GPKG
    if "hotspots_gdf" in hotspots_results:
        gpkg_path = output_path / "hotspots_deforestation.gpkg"
        hotspots_results["hotspots_gdf"].to_file(gpkg_path, driver="GPKG")
        saved["hotspots_gpkg"] = str(gpkg_path)
        logger.info(f"Hotspots guardados: {gpkg_path}")
    
    # Gi* raster → GeoTIFF
    if "gi_raster" in hotspots_results and hotspots_results["gi_raster"] is not None:
        # Requiere template para transform/crs - se maneja en export/raster_export
        pass
    
    # Resumen → CSV
    if "summary" in hotspots_results:
        summary_df = pd.DataFrame([hotspots_results["summary"]])
        csv_path = output_path / "hotspots_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        saved["summary_csv"] = str(csv_path)
    
    # Por transición
    if "by_transition" in hotspots_results.get("summary", {}):
        trans_df = pd.DataFrame(hotspots_results["summary"]["by_transition"])
        trans_csv = output_path / "hotspots_by_transition.csv"
        trans_df.to_csv(trans_csv, index=False)
        saved["by_transition_csv"] = str(trans_csv)
    
    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if HAS_PYSAL:
        print("Dependencias OK: libpysal, esda disponibles")
    else:
        print("FALTAN dependencias: conda install -c conda-forge libpysal esda")