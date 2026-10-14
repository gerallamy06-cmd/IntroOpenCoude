"""
Módulo Detección de Cambio: Matrices de transición anuales + global + raster de transiciones
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import xarray as xr
import pandas as pd
from scipy import sparse
from collections import Counter

logger = logging.getLogger(__name__)

# Clases MapBiomas (1-7, 0=NoData)
MAPBIOMAS_CLASSES = [1, 2, 3, 4, 5, 6, 7]
N_CLASSES = len(MAPBIOMAS_CLASSES)

# Mapeo código transición: from * 10 + to
# ej: 12 = Bosque(1) → Pastizales(2), 14 = Bosque(1) → Urbano(4)
TRANSITION_CODES = {
    (f, t): f * 10 + t
    for f in MAPBIOMAS_CLASSES
    for t in MAPBIOMAS_CLASSES
    if f != t
}

# Transiciones de interés para análisis (deforestación, urbanización)
KEY_TRANSITIONS = {
    "deforestacion_pastizales": (1, 2),   # Bosque → Pastizales
    "deforestacion_cultivos": (1, 3),     # Bosque → Cultivos
    "deforestacion_urbano": (1, 4),       # Bosque → Urbano
    "pastizales_urbano": (2, 4),          # Pastizales → Urbano
    "cultivos_urbano": (3, 4),            # Cultivos → Urbano
    "reforestacion": (2, 1),              # Pastizales → Bosque
    "reforestacion_cultivos": (3, 1),     # Cultivos → Bosque
    "humedales_perdida": (7, 2),          # Humedales → Pastizales
    "agua_cambio": (5, 2),                # Agua → Pastizales (secamiento)
}


def compute_transition_matrix(
    year1_data: np.ndarray,
    year2_data: np.ndarray,
    classes: List[int] = None,
    nodata_value: int = 0
) -> Tuple[np.ndarray, Dict]:
    """
    Calcula matriz de transición entre dos años.
    
    Args:
        year1_data: Array (H, W) - año inicial
        year2_data: Array (H, W) - año final
        classes: Lista de códigos de clase válidos
        nodata_value: Valor NoData
    
    Returns:
        (matrix, metadata)
        matrix: (n_classes, n_classes) - conteos de píxeles
        metadata: dict con info adicional
    """
    if classes is None:
        classes = MAPBIOMAS_CLASSES
    
    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    # Máscara de píxeles válidos en ambos años
    valid_mask = (year1_data != nodata_value) & (year2_data != nodata_value)
    
    if not valid_mask.any():
        logger.warning("No hay píxeles válidos en común")
        return np.zeros((n, n), dtype=np.int64), {"valid_pixels": 0}
    
    y1_valid = year1_data[valid_mask]
    y2_valid = year2_data[valid_mask]
    
    # Filtrar solo clases válidas
    class_mask = np.isin(y1_valid, classes) & np.isin(y2_valid, classes)
    y1_valid = y1_valid[class_mask]
    y2_valid = y2_valid[class_mask]
    
    if len(y1_valid) == 0:
        return np.zeros((n, n), dtype=np.int64), {"valid_pixels": 0}
    
    # Convertir a índices
    y1_idx = np.array([class_to_idx[v] for v in y1_valid])
    y2_idx = np.array([class_to_idx[v] for v in y2_valid])
    
    # Calcular matriz usando bincount 2D (rápido)
    # idx = y1_idx * n + y2_idx
    flat_idx = y1_idx * n + y2_idx
    counts = np.bincount(flat_idx, minlength=n * n)
    matrix = counts.reshape((n, n)).astype(np.int64)
    
    # Metadatos
    metadata = {
        "valid_pixels": int(valid_mask.sum()),
        "class_pixels_year1": int((year1_data != nodata_value).sum()),
        "class_pixels_year2": int((year2_data != nodata_value).sum()),
        "changed_pixels": int((y1_valid != y2_valid).sum()),
        "unchanged_pixels": int((y1_valid == y2_valid).sum()),
        "change_rate_pct": round((y1_valid != y2_valid).sum() / len(y1_valid) * 100, 2) if len(y1_valid) > 0 else 0
    }
    
    return matrix, metadata


def matrix_to_dataframe(
    matrix: np.ndarray,
    classes: List[int] = None,
    year_from: int = None,
    year_to: int = None
) -> pd.DataFrame:
    """Convierte matriz de transición a DataFrame legible."""
    if classes is None:
        classes = MAPBIOMAS_CLASSES
    
    rows = []
    for i, from_cls in enumerate(classes):
        for j, to_cls in enumerate(classes):
            count = matrix[i, j]
            if count > 0:
                rows.append({
                    "year_from": year_from,
                    "year_to": year_to,
                    "from_class": from_cls,
                    "from_class_name": get_class_name(from_cls),
                    "to_class": to_cls,
                    "to_class_name": get_class_name(to_cls),
                    "pixel_count": int(count),
                    "is_change": from_cls != to_cls
                })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pixel_count", ascending=False).reset_index(drop=True)
    return df


def get_class_name(code: int) -> str:
    """Nombre de clase MapBiomas."""
    names = {
        1: "Bosque", 2: "Pastizales", 3: "Cultivos",
        4: "Urbano", 5: "Agua", 6: "Suelo desnudo", 7: "Humedales"
    }
    return names.get(code, f"Clase_{code}")


def compute_annual_transitions(
    stack: xr.DataArray,
    years: List[int],
    classes: List[int] = None,
    nodata_value: int = 0
) -> Dict[str, Any]:
    """
    Calcula matrices de transición para pares anuales consecutivos.
    
    Args:
        stack: DataArray (time, band, y, x) remapeado a MapBiomas
        years: Lista de años [2017, 2018, ...]
        classes: Clases válidas
        nodata_value: Valor NoData
    
    Returns:
        Dict con matrices, dataframes, metadatos
    """
    if classes is None:
        classes = MAPBIOMAS_CLASSES
    
    logger.info(f"Calculando transiciones anuales para {len(years)-1} pares...")
    
    # Convertir a numpy (time, y, x)
    data = stack.isel(band=0).data
    if hasattr(data, 'compute'):
        logger.info("Computando stack a numpy...")
        data = data.compute()
    
    # data shape: (n_time, H, W)
    n_time = data.shape[0]
    
    results = {
        "matrices": {},
        "dataframes": {},
        "metadata": {},
        "transition_rasters": {}
    }
    
    for i in range(n_time - 1):
        year_from = years[i]
        year_to = years[i + 1]
        pair_key = f"{year_from}_{year_to}"
        
        logger.info(f"Procesando {pair_key}...")
        
        y1_data = data[i]   # (H, W)
        y2_data = data[i + 1]
        
        # Matriz de transición
        matrix, meta = compute_transition_matrix(
            y1_data, y2_data, classes, nodata_value
        )
        
        # DataFrame
        df = matrix_to_dataframe(matrix, classes, year_from, year_to)
        
        # Raster de transiciones (código from*10+to)
        transition_raster = create_transition_raster(y1_data, y2_data, nodata_value)
        
        results["matrices"][pair_key] = matrix
        results["dataframes"][pair_key] = df
        results["metadata"][pair_key] = meta
        results["transition_rasters"][pair_key] = transition_raster
        
        logger.info(f"  {pair_key}: {meta['changed_pixels']:,} píxeles cambiados "
                    f"({meta['change_rate_pct']:.2f}%)")
    
    return results


def create_transition_raster(
    year1_data: np.ndarray,
    year2_data: np.ndarray,
    nodata_value: int = 0
) -> np.ndarray:
    """
    Crea raster de códigos de transición: from*10 + to.
    Valores:
    - 0: NoData o sin cambio (from==to)
    - 11-77: código transición (ej 12=Bosque→Pastizales)
    """
    # Máscara válida
    valid = (year1_data != nodata_value) & (year2_data != nodata_value)
    
    # Inicializar con 0
    transition = np.zeros_like(year1_data, dtype=np.uint8)
    
    # Solo donde hay cambio
    changed = valid & (year1_data != year2_data)
    
    if changed.any():
        from_vals = year1_data[changed]
        to_vals = year2_data[changed]
        
        # Calcular código: from*10 + to
        codes = from_vals * 10 + to_vals
        transition[changed] = codes.astype(np.uint8)
    
    return transition


def compute_global_transition(
    stack: xr.DataArray,
    years: List[int],
    year_from: int,
    year_to: int,
    classes: List[int] = None,
    nodata_value: int = 0
) -> Dict[str, Any]:
    """
    Calcula matriz de transición global entre año inicial y final.
    """
    logger.info(f"Calculando transición global {year_from} → {year_to}...")
    
    data = stack.isel(band=0).data
    if hasattr(data, 'compute'):
        data = data.compute()
    
    # Encontrar índices de años
    try:
        idx_from = years.index(year_from)
        idx_to = years.index(year_to)
    except ValueError as e:
        raise ValueError(f"Año no encontrado en datos: {e}")
    
    y1_data = data[idx_from]
    y2_data = data[idx_to]
    
    matrix, meta = compute_transition_matrix(
        y1_data, y2_data, classes, nodata_value
    )
    
    df = matrix_to_dataframe(matrix, classes, year_from, year_to)
    transition_raster = create_transition_raster(y1_data, y2_data, nodata_value)
    
    return {
        "matrix": matrix,
        "dataframe": df,
        "metadata": meta,
        "transition_raster": transition_raster,
        "years": (year_from, year_to)
    }


def calculate_net_change(
    matrix: np.ndarray,
    classes: List[int] = None,
    pixel_area_ha: float = 0.01
) -> pd.DataFrame:
    """
    Calcula cambio neto por clase (ganancia - pérdida) en hectáreas.
    """
    if classes is None:
        classes = MAPBIOMAS_CLASSES
    
    n = len(classes)
    
    # Ganancia: suma de columna (llegadas a la clase) - diagonal
    gains = matrix.sum(axis=0) - np.diag(matrix)
    # Pérdida: suma de fila (salidas de la clase) - diagonal
    losses = matrix.sum(axis=1) - np.diag(matrix)
    # Neto
    net = gains - losses
    
    rows = []
    for i, cls in enumerate(classes):
        rows.append({
            "class_code": cls,
            "class_name": get_class_name(cls),
            "gain_pixels": int(gains[i]),
            "loss_pixels": int(losses[i]),
            "net_pixels": int(net[i]),
            "gain_ha": round(gains[i] * pixel_area_ha, 2),
            "loss_ha": round(losses[i] * pixel_area_ha, 2),
            "net_ha": round(net[i] * pixel_area_ha, 2)
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("net_ha", ascending=False).reset_index(drop=True)
    return df


def get_dominant_transitions(
    df: pd.DataFrame,
    top_n: int = 10,
    exclude_unchanged: bool = True
) -> pd.DataFrame:
    """Obtiene las transiciones dominantes ordenadas por píxeles."""
    if exclude_unchanged:
        df = df[df["is_change"]].copy()
    return df.head(top_n).reset_index(drop=True)


def calculate_annual_change_rates(
    annual_results: Dict,
    pixel_area_ha: float = 0.01
) -> pd.DataFrame:
    """
    Calcula tasas de cambio anuales por clase.
    """
    rows = []
    
    for pair_key, meta in annual_results["metadata"].items():
        year_from, year_to = pair_key.split("_")
        
        # Cambio total
        rows.append({
            "period": pair_key,
            "year_from": int(year_from),
            "year_to": int(year_to),
            "changed_pixels": meta["changed_pixels"],
            "changed_ha": round(meta["changed_pixels"] * pixel_area_ha, 2),
            "change_rate_pct": meta["change_rate_pct"],
            "valid_pixels": meta["valid_pixels"]
        })
    
    return pd.DataFrame(rows)


def create_change_summary(
    annual_results: Dict,
    global_result: Dict,
    years: List[int],
    pixel_area_ha: float = 0.01
) -> Dict[str, pd.DataFrame]:
    """
    Crea resumen completo de cambio.
    """
    summary = {}
    
    # 1. Cambio neto global
    summary["net_change_global"] = calculate_net_change(
        global_result["matrix"], pixel_area_ha=pixel_area_ha
    )
    
    # 2. Cambio neto por período anual
    for pair_key, matrix in annual_results["matrices"].items():
        summary[f"net_change_{pair_key}"] = calculate_net_change(
            matrix, pixel_area_ha=pixel_area_ha
        )
    
    # 3. Transiciones dominantes globales
    summary["dominant_transitions_global"] = get_dominant_transitions(
        global_result["dataframe"], top_n=15
    )
    
    # 4. Transiciones dominantes por período
    for pair_key, df in annual_results["dataframes"].items():
        summary[f"dominant_transitions_{pair_key}"] = get_dominant_transitions(
            df, top_n=10
        )
    
    # 5. Tasas de cambio anual
    summary["annual_change_rates"] = calculate_annual_change_rates(
        annual_results, pixel_area_ha
    )
    
    # 6. Resumen ejecutivo
    exec_rows = []
    for pair_key, meta in annual_results["metadata"].items():
        exec_rows.append({
            "period": pair_key,
            "total_changed_ha": round(meta["changed_pixels"] * pixel_area_ha, 2),
            "change_rate_pct": meta["change_rate_pct"],
            "valid_pixels": meta["valid_pixels"]
        })
    
    # Global
    global_meta = global_result["metadata"]
    exec_rows.append({
        "period": f"{years[0]}_{years[-1]}",
        "total_changed_ha": round(global_meta["changed_pixels"] * pixel_area_ha, 2),
        "change_rate_pct": global_meta["change_rate_pct"],
        "valid_pixels": global_meta["valid_pixels"]
    })
    
    summary["executive_summary"] = pd.DataFrame(exec_rows)
    
    return summary


def save_transition_results(
    annual_results: Dict,
    global_result: Dict,
    output_dir: str,
    years: List[int],
    template_stack: xr.DataArray = None
) -> Dict[str, str]:
    """
    Guarda todos los resultados de cambio a disco.
    Retorna dict con paths guardados.
    """
    from pathlib import Path
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved = {}
    
    # Guardar matrices como CSV
    for pair_key, matrix in annual_results["matrices"].items():
        df = annual_results["dataframes"][pair_key]
        csv_path = output_path / f"transition_matrix_{pair_key}.csv"
        df.to_csv(csv_path, index=False)
        saved[f"matrix_{pair_key}"] = str(csv_path)
    
    # Matriz global
    global_df = global_result["dataframe"]
    global_csv = output_path / "transition_matrix_global.csv"
    global_df.to_csv(global_csv, index=False)
    saved["matrix_global"] = str(global_csv)
    
    # Guardar metadatos como JSON
    meta_all = {}
    for pair_key, meta in annual_results["metadata"].items():
        meta_all[pair_key] = meta
    meta_all["global"] = global_result["metadata"]
    
    meta_json = output_path / "transition_metadata.json"
    with open(meta_json, 'w') as f:
        json.dump(meta_all, f, indent=2, default=str)
    saved["metadata"] = str(meta_json)
    
    logger.info(f"Resultados de cambio guardados en: {output_path}")
    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test rápido
    y1 = np.array([[1, 1, 2, 3], [1, 2, 2, 4], [5, 5, 6, 7]], dtype=np.uint8)
    y2 = np.array([[1, 2, 2, 3], [2, 2, 3, 4], [5, 2, 6, 7]], dtype=np.uint8)
    
    matrix, meta = compute_transition_matrix(y1, y2)
    print("Matriz:")
    print(matrix)
    print(f"Meta: {meta}")
    
    df = matrix_to_dataframe(matrix, year_from=2017, year_to=2018)
    print("\nDataFrame:")
    print(df)
    
    net = calculate_net_change(matrix)
    print("\nCambio neto:")
    print(net)