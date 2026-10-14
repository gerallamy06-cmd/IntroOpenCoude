"""
Módulo Remapeo de Clases: IO-LULC 9 clases → MapBiomas Colombia 7 clases
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Mapeo IO-LULC (Esri 10m Annual v02) → MapBiomas Colombia simplificado
# IO-LULC values: 0=NoData, 1=Water, 2=Trees, 3=Grass, 4=Flooded Veg, 5=Crops, 6=Scrub, 7=Built, 8=Bare, 9=Snow, 10=Clouds, 11=Rangeland
# MapBiomas: 1=Bosque, 2=Pastizales, 3=Cultivos, 4=Urbano, 5=Agua, 6=Suelo desnudo, 7=Humedales, 0=NoData

DEFAULT_REMAP = {
    1: 5,   # Water → Agua
    2: 1,   # Trees → Bosque
    3: 2,   # Grass → Pastizales (opcional, no en IO-LULC v02 principal)
    4: 7,   # Flooded Vegetation → Humedales
    5: 3,   # Crops → Cultivos
    6: 2,   # Scrub/Shrub → Pastizales (vegetación arbustiva)
    7: 4,   # Built Area → Urbano
    8: 6,   # Bare Ground → Suelo desnudo
    9: 0,   # Snow/Ice → NoData
    10: 0,  # Clouds → NoData
    11: 2   # Rangeland → Pastizales
}

# Nombres de clases MapBiomas
MAPBIOMAS_CLASS_NAMES = {
    0: "No Data",
    1: "Bosque",
    2: "Pastizales",
    3: "Cultivos",
    4: "Urbano",
    5: "Agua",
    6: "Suelo desnudo",
    7: "Humedales"
}

# Nombres de clases IO-LULC
IO_LULC_CLASS_NAMES = {
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


def build_remap_lookup(remap_dict: Dict[int, int], max_value: int = 255) -> np.ndarray:
    """
    Construye array de lookup para remapeo vectorizado rápido.
    
    Args:
        remap_dict: Diccionario {valor_original: valor_nuevo}
        max_value: Valor máximo esperado en datos (para tamaño array)
    
    Returns:
        numpy array donde lookup[valor_original] = valor_nuevo
    """
    lookup = np.zeros(max_value + 1, dtype=np.uint8)
    for src_val, dst_val in remap_dict.items():
        if src_val <= max_value:
            lookup[src_val] = dst_val
    return lookup


def remap_classes_numpy(
    data: np.ndarray,
    remap_dict: Dict[int, int],
    nodata_values: Optional[List[int]] = None,
    inplace: bool = False
) -> np.ndarray:
    """
    Remapea clases en array numpy usando lookup vectorizado.
    
    Args:
        data: Array de entrada (cualquier forma)
        remap_dict: Diccionario mapeo {original: nuevo}
        nodata_values: Valores a forzar a 0 (NoData)
        inplace: Modificar array original
    
    Returns:
        Array remapeado
    """
    if not inplace:
        data = data.copy()
    
    # Crear lookup
    max_val = data.max() if data.size > 0 else 255
    lookup = build_remap_lookup(remap_dict, max_val)
    
    # Aplicar remapeo vectorizado
    original_shape = data.shape
    flat_data = data.ravel()
    remapped_flat = lookup[flat_data]
    data = remapped_flat.reshape(original_shape)
    
    # Forzar nodata values a 0
    if nodata_values:
        for nodata_val in nodata_values:
            data[data == remap_dict.get(nodata_val, nodata_val)] = 0
    
    return data


def remap_stack(
    stack: xr.DataArray,
    remap_config: Dict,
    band: int = 0
) -> xr.DataArray:
    """
    Remapea clases en stack xarray completo (time, band, y, x).
    
    Args:
        stack: DataArray con datos LULC
        remap_config: Config con 'remap' dict y 'nodata_values' list
        band: Índice de banda a remapear (default 0)
    
    Returns:
        DataArray remapeado con mismas coordenadas y atributos
    """
    remap_dict = remap_config.get("remap", DEFAULT_REMAP)
    nodata_values = remap_config.get("nodata_values", [0, 9, 10])
    
    logger.info(f"Remapeando clases: {len(remap_dict)} mapeos definidos")
    logger.debug(f"Mapeo: {remap_dict}")
    logger.debug(f"NoData values: {nodata_values}")
    
    # Extraer banda a remapear
    data = stack.isel(band=band)
    
    # Convertir a numpy (compute si dask)
    if hasattr(data.data, 'compute'):
        logger.info("Computando stack para remapeo...")
        np_data = data.data.compute()
    else:
        np_data = data.data
    
    # np_data shape: (time, y, x)
    n_time = np_data.shape[0]
    remapped_frames = []
    
    # Procesar cada time step
    for t in range(n_time):
        frame = np_data[t]  # (y, x)
        remapped = remap_classes_numpy(frame, remap_dict, nodata_values)
        remapped_frames.append(remapped)
    
    remapped_data = np.stack(remapped_frames, axis=0)  # (time, y, x)
    remapped_data = remapped_data[:, np.newaxis, :, :]  # (time, 1, y, x)
    
    # Crear nuevo DataArray
    new_stack = xr.DataArray(
        remapped_data,
        dims=["time", "band", "y", "x"],
        coords={
            "time": stack.time.values,
            "band": [f"mapbiomas_class"],  # Nueva banda
            "y": stack.y.values,
            "x": stack.x.values
        },
        attrs=stack.attrs.copy()
    )
    
    # Copiar atributos rio
    if hasattr(stack, 'rio'):
        new_stack.rio.write_crs(stack.rio.crs, inplace=True)
        new_stack.rio.write_transform(stack.rio.transform(), inplace=True)
        new_stack.rio.write_nodata(0, inplace=True)  # 0 = NoData en MapBiomas
    
    # Añadir metadatos de clases
    new_stack.attrs["class_mapping"] = str(remap_dict)
    new_stack.attrs["class_names"] = str(MAPBIOMAS_CLASS_NAMES)
    new_stack.attrs["source_classes"] = str(IO_LULC_CLASS_NAMES)
    
    logger.info(f"Remapeo completado. Shape: {new_stack.shape}")
    logger.info(f"Clases resultantes: {sorted(set(np.unique(remapped_data)) - {0})}")
    
    return new_stack


def validate_remap(remapped_stack: xr.DataArray, expected_classes: List[int] = None) -> Dict:
    """
    Valida el remapeo: verifica clases presentes, conteos, etc.
    """
    if expected_classes is None:
        expected_classes = [1, 2, 3, 4, 5, 6, 7]
    
    data = remapped_stack.isel(band=0).data
    if hasattr(data, 'compute'):
        data = data.compute()
    
    unique, counts = np.unique(data, return_counts=True)
    
    validation = {
        "classes_found": unique.tolist(),
        "class_counts": dict(zip(unique.tolist(), counts.tolist())),
        "total_pixels": int(data.size),
        "valid_pixels": int((data != 0).sum()),
        "nodata_pixels": int((data == 0).sum()),
        "expected_classes_present": [c for c in expected_classes if c in unique],
        "expected_classes_missing": [c for c in expected_classes if c not in unique],
        "unexpected_classes": [c for c in unique if c not in expected_classes and c != 0]
    }
    
    logger.info(f"Validación remapeo: {validation['valid_pixels']:,} píxeles válidos, "
                f"{validation['nodata_pixels']:,} NoData")
    logger.info(f"Clases presentes: {validation['classes_found']}")
    
    if validation['expected_classes_missing']:
        logger.warning(f"Clases esperadas ausentes: {validation['expected_classes_missing']}")
    
    return validation


def create_class_lookup_table(remap_dict: Dict[int, int]) -> Dict:
    """
    Crea tabla de lookup completa para exportación a GeoTIFF (GDAL LUT).
    """
    # GDAL LUT requiere entradas para todos los valores 0-255
    lut = {}
    for i in range(256):
        lut[i] = remap_dict.get(i, 0)  # Default a 0 (NoData)
    
    return lut


def get_class_statistics_per_year(
    remapped_stack: xr.DataArray,
    pixel_area_ha: float = 0.01  # 10m x 10m = 0.01 ha
) -> Dict:
    """
    Calcula estadísticas de área por clase y año.
    
    Returns:
        Dict con DataFrame pandas por año
    """
    import pandas as pd
    
    data = remapped_stack.isel(band=0).data
    if hasattr(data, 'compute'):
        data = data.compute()
    
    n_time = data.shape[0]
    years = [str(t)[:4] for t in remapped_stack.time.values[:n_time]]
    
    results = {}
    for t in range(n_time):
        frame = data[t]
        unique, counts = np.unique(frame, return_counts=True)
        
        year_stats = []
        for cls_val, count in zip(unique, counts):
            if cls_val == 0:
                continue  # Skip NoData
            year_stats.append({
                "year": years[t],
                "class_code": int(cls_val),
                "class_name": MAPBIOMAS_CLASS_NAMES.get(int(cls_val), f"Class_{cls_val}"),
                "pixel_count": int(count),
                "area_ha": round(count * pixel_area_ha, 2),
                "area_pct": round(count / (frame != 0).sum() * 100, 2) if (frame != 0).sum() > 0 else 0
            })
        
        results[years[t]] = pd.DataFrame(year_stats)
    
    return results


def print_class_summary(remapped_stack: xr.DataArray, pixel_area_ha: float = 0.01):
    """Imprime resumen de clases por año."""
    stats = get_class_statistics_per_year(remapped_stack, pixel_area_ha)
    
    for year, df in stats.items():
        print(f"\n=== {year} ===")
        print(df.to_string(index=False))
        total_ha = df["area_ha"].sum()
        print(f"Total área clasificada: {total_ha:,.2f} ha")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test rápido
    test_data = np.array([[1, 2, 5, 7, 11, 0, 9, 10]], dtype=np.uint8)
    remapped = remap_classes_numpy(test_data, DEFAULT_REMAP, [0, 9, 10])
    print(f"Original: {test_data}")
    print(f"Remapeado: {remapped}")
    print(f"Clases MapBiomas: {MAPBIOMAS_CLASS_NAMES}")