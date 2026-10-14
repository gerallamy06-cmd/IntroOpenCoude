"""
Módulo Estadísticas: Cálculo de áreas, resúmenes, dataframes consolidados
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
import json

logger = logging.getLogger(__name__)

# Clases MapBiomas
MAPBIOMAS_CLASSES = [1, 2, 3, 4, 5, 6, 7]
CLASS_NAMES = {
    1: "Bosque", 2: "Pastizales", 3: "Cultivos",
    4: "Urbano", 5: "Agua", 6: "Suelo desnudo", 7: "Humedales"
}


def calculate_pixel_area_ha(transform, crs: str = "EPSG:4326") -> float:
    """
    Calcula área de un píxel en hectáreas.
    Si CRS es geográfico (EPSG:4326), usa aproximación latitudinal.
    Si CRS es proyectado (ej. EPSG:3116), usa transform directamente.
    """
    if crs and "4326" in str(crs):
        # Aproximación: 1° lat ≈ 111 km, 1° lon ≈ 111*cos(lat) km
        # Para Colombia (≈4°N), cos(4°) ≈ 0.997
        pixel_width_deg = abs(transform[0])
        pixel_height_deg = abs(transform[4])
        # Convertir a metros
        pixel_width_m = pixel_width_deg * 111000 * 0.997  # ajustado por latitud
        pixel_height_m = pixel_height_deg * 111000
        area_m2 = pixel_width_m * pixel_height_m
        return area_m2 / 10000  # hectáreas
    else:
        # CRS proyectado: usar transform directamente (unidades en metros)
        pixel_width_m = abs(transform[0])
        pixel_height_m = abs(transform[4])
        area_m2 = pixel_width_m * pixel_height_m
        return area_m2 / 10000


def get_class_statistics(
    stack: xr.DataArray,
    pixel_area_ha: float = None,
    nodata_value: int = 0,
    classes: List[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Calcula estadísticas de área por clase y año.
    
    Args:
        stack: DataArray (time, band, y, x) remapeado a MapBiomas
        pixel_area_ha: Área por píxel en hectáreas (None = calcular desde transform)
        nodata_value: Valor NoData
        classes: Lista de clases válidas
    
    Returns:
        Dict con DataFrames: 'by_year', 'by_class', 'summary'
    """
    if classes is None:
        classes = MAPBIOMAS_CLASSES
    
    # Calcular pixel_area si no se proporciona
    if pixel_area_ha is None:
        if hasattr(stack, 'rio'):
            transform = stack.rio.transform()
            crs = str(stack.rio.crs)
            pixel_area_ha = calculate_pixel_area_ha(transform, crs)
        else:
            pixel_area_ha = 0.01  # default 10m x 10m
        logger.info(f"Área por píxel calculada: {pixel_area_ha:.6f} ha")
    
    # Extraer datos (time, y, x)
    data = stack.isel(band=0).data
    if hasattr(data, 'compute'):
        logger.info("Computando stack para estadísticas...")
        data = data.compute()
    
    n_time = data.shape[0]
    years = [str(t)[:4] for t in stack.time.values[:n_time]]
    
    # Resultados por año
    yearly_rows = []
    
    for t in range(n_time):
        year = years[t]
        frame = data[t]  # (H, W)
        
        # Conteo por clase
        unique, counts = np.unique(frame, return_counts=True)
        
        for cls_val, count in zip(unique, counts):
            if cls_val == nodata_value:
                continue
            
            yearly_rows.append({
                "year": year,
                "class_code": int(cls_val),
                "class_name": CLASS_NAMES.get(int(cls_val), f"Class_{cls_val}"),
                "pixel_count": int(count),
                "area_ha": round(count * pixel_area_ha, 2),
                "area_km2": round(count * pixel_area_ha / 100, 4)
            })
    
    df_yearly = pd.DataFrame(yearly_rows)
    
    # Porcentaje por año
    for year in years:
        year_mask = df_yearly["year"] == year
        total_ha = df_yearly.loc[year_mask, "area_ha"].sum()
        if total_ha > 0:
            df_yearly.loc[year_mask, "pct_of_total"] = (
                df_yearly.loc[year_mask, "area_ha"] / total_ha * 100
            ).round(2)
    
    # Resumen por clase (promedio temporal)
    class_summary_rows = []
    for cls in classes:
        cls_data = df_yearly[df_yearly["class_code"] == cls]
        if not cls_data.empty:
            class_summary_rows.append({
                "class_code": cls,
                "class_name": CLASS_NAMES.get(cls, f"Class_{cls}"),
                "mean_area_ha": round(cls_data["area_ha"].mean(), 2),
                "std_area_ha": round(cls_data["area_ha"].std(), 2),
                "min_area_ha": round(cls_data["area_ha"].min(), 2),
                "max_area_ha": round(cls_data["area_ha"].max(), 2),
                "trend_ha_per_year": round(
                    np.polyfit(range(len(cls_data)), cls_data["area_ha"], 1)[0], 2
                ) if len(cls_data) > 1 else 0
            })
    
    df_class_summary = pd.DataFrame(class_summary_rows)
    
    # Resumen general
    total_area_by_year = df_yearly.groupby("year")["area_ha"].sum().reset_index()
    total_area_by_year.columns = ["year", "total_area_ha"]
    
    return {
        "by_year": df_yearly,
        "by_class": df_class_summary,
        "total_by_year": total_area_by_year,
        "pixel_area_ha": pixel_area_ha
    }


def calculate_transition_statistics(
    change_results: Dict,
    pixel_area_ha: float = 0.01
) -> Dict[str, pd.DataFrame]:
    """
    Calcula estadísticas detalladas de transiciones.
    """
    from .change_detection import calculate_net_change, get_dominant_transitions
    
    stats = {}
    
    # Cambio neto global
    if "global" in change_results:
        global_matrix = change_results["global"]["matrix"]
        stats["net_change_global"] = calculate_net_change(
            global_matrix, pixel_area_ha=pixel_area_ha
        )
        
        stats["dominant_transitions_global"] = get_dominant_transitions(
            change_results["global"]["dataframe"], top_n=20
        )
    
    # Cambio neto anual
    if "annual" in change_results:
        for pair_key, matrix in change_results["annual"]["matrices"].items():
            stats[f"net_change_{pair_key}"] = calculate_net_change(
                matrix, pixel_area_ha=pixel_area_ha
            )
            
            stats[f"dominant_transitions_{pair_key}"] = get_dominant_transitions(
                change_results["annual"]["dataframes"][pair_key], top_n=10
            )
    
    return stats


def create_consolidated_report(
    area_stats: Dict,
    transition_stats: Dict,
    hotspots_stats: Dict = None,
    config: Dict = None
) -> Dict[str, Any]:
    """
    Crea reporte consolidado en formato dict para JSON.
    """
    report = {
        "metadata": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "config": config or {},
            "mapbiomas_classes": CLASS_NAMES
        },
        "area_statistics": {
            "by_year": area_stats["by_year"].to_dict("records"),
            "by_class": area_stats["by_class"].to_dict("records"),
            "total_by_year": area_stats["total_by_year"].to_dict("records"),
            "pixel_area_ha": area_stats.get("pixel_area_ha", 0.01)
        },
        "transition_statistics": {}
    }
    
    # Añadir estadísticas de transición
    for key, df in transition_stats.items():
        if isinstance(df, pd.DataFrame):
            report["transition_statistics"][key] = df.to_dict("records")
    
    # Añadir hotspots si disponibles
    if hotspots_stats:
        report["hotspots"] = hotspots_stats
    
    return report


def save_statistics(
    area_stats: Dict,
    transition_stats: Dict,
    hotspots_stats: Dict = None,
    output_dir: str = "output/stats"
) -> Dict[str, str]:
    """
    Guarda todas las estadísticas a CSV y JSON.
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved = {}
    
    # Area stats
    for key, df in area_stats.items():
        if isinstance(df, pd.DataFrame):
            csv_path = output_path / f"area_{key}.csv"
            df.to_csv(csv_path, index=False)
            saved[f"area_{key}"] = str(csv_path)
    
    # Transition stats
    for key, df in transition_stats.items():
        if isinstance(df, pd.DataFrame):
            csv_path = output_path / f"transition_{key}.csv"
            df.to_csv(csv_path, index=False)
            saved[f"transition_{key}"] = str(csv_path)
    
    # Consolidated JSON
    report = create_consolidated_report(area_stats, transition_stats, hotspots_stats)
    json_path = output_path / "consolidated_report.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    saved["consolidated_json"] = str(json_path)
    
    logger.info(f"Estadísticas guardadas en: {output_path}")
    return saved


def print_summary(area_stats: Dict, transition_stats: Dict):
    """Imprime resumen legible en consola."""
    print("\n" + "="*60)
    print("RESUMEN ESTADÍSTICO COBERTURA TERRESTRE REGIÓN ANDINA")
    print("="*60)
    
    # Total por año
    print("\n📊 ÁREA TOTAL POR AÑO (hectáreas):")
    total_df = area_stats["total_by_year"]
    for _, row in total_df.iterrows():
        print(f"   {row['year']}: {row['total_area_ha']:>12,.2f} ha")
    
    # Promedio por clase
    print("\n🌳 PROMEDIO TEMPORAL POR CLASE:")
    class_df = area_stats["by_class"]
    for _, row in class_df.iterrows():
        trend = "📈" if row['trend_ha_per_year'] > 0 else "📉" if row['trend_ha_per_year'] < 0 else "➡️"
        print(f"   {row['class_name']:15s}: {row['mean_area_ha']:>10,.2f} ha ({trend} {row['trend_ha_per_year']:+.2f} ha/año)")
    
    # Cambio neto global
    if "net_change_global" in transition_stats:
        print("\n🔄 CAMBIO NETO GLOBAL (ha):")
        net_df = transition_stats["net_change_global"]
        for _, row in net_df.iterrows():
            sign = "+" if row['net_ha'] > 0 else ""
            print(f"   {row['class_name']:15s}: {sign}{row['net_ha']:>8.2f} ha "
                  f"(ganancia: {row['gain_ha']:.2f}, pérdida: {row['loss_ha']:.2f})")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test rápido
    test_data = np.random.randint(0, 8, (3, 100, 100), dtype=np.uint8)
    test_data[0, 0, 0] = 0  # NoData
    
    import xarray as xr
    stack = xr.DataArray(
        test_data[:, np.newaxis, :, :],
        dims=["time", "band", "y", "x"],
        coords={"time": ["2017", "2018", "2019"]}
    )
    
    stats = get_class_statistics(stack)
    print_summary(stats, {})