"""
Módulo Colormaps: MapBiomas Colombia oficiales + Hotspots Getis-Ord Gi*
"""
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

logger = logging.getLogger(__name__)

# ============================================================
# COLORMAP MAPBIOMAS COLOMBIA (Oficial)
# ============================================================

MAPBIOMAS_COLORS = {
    0: "#00000000",  # Transparente (NoData)
    1: "#129912",    # Bosque - Verde oscuro
    2: "#FFFFB2",    # Pastizales - Amarillo claro
    3: "#E8C16F",    # Cultivos - Naranja claro
    4: "#E86B4C",    # Urbano - Rojo ladrillo
    5: "#4C9CE8",    # Agua - Azul
    6: "#D68B4C",    # Suelo desnudo - Marrón
    7: "#A6D8F0",    # Humedales - Azul claro
}

MAPBIOMAS_CLASS_NAMES = {
    0: "Sin datos",
    1: "Bosque",
    2: "Pastizales",
    3: "Cultivos",
    4: "Urbano",
    5: "Agua",
    6: "Suelo desnudo",
    7: "Humedales"
}

# Orden para array de colormap (0-7)
MAPBIOMAS_COLOR_ARRAY = [
    MAPBIOMAS_COLORS[0],  # 0: transparente
    MAPBIOMAS_COLORS[1],  # 1: Bosque
    MAPBIOMAS_COLORS[2],  # 2: Pastizales
    MAPBIOMAS_COLORS[3],  # 3: Cultivos
    MAPBIOMAS_COLORS[4],  # 4: Urbano
    MAPBIOMAS_COLORS[5],  # 5: Agua
    MAPBIOMAS_COLORS[6],  # 6: Suelo desnudo
    MAPBIOMAS_COLORS[7],  # 7: Humedales
]

# ============================================================
# COLORMAP HOTSPOTS GETIS-ORD Gi*
# ============================================================

HOTSPOT_COLORS = {
    -3: "#00008B",  # Cold Spot 99.9% - Azul oscuro
    -2: "#0000FF",  # Cold Spot 99%   - Azul
    -1: "#4169E1",  # Cold Spot 95%   - Royal Blue
    0:  "#CCCCCC",  # No significativo - Gris
    1:  "#FFD700",  # Hot Spot 95%    - Dorado
    2:  "#FF8C00",  # Hot Spot 99%    - Naranja oscuro
    3:  "#FF0000",  # Hot Spot 99.9%  - Rojo
}

HOTSPOT_LABELS = {
    -3: "Cold Spot 99.9%",
    -2: "Cold Spot 99%",
    -1: "Cold Spot 95%",
    0:  "No significativo",
    1:  "Hot Spot 95%",
    2:  "Hot Spot 99%",
    3:  "Hot Spot 99.9%"
}

HOTSPOT_COLOR_ARRAY = [
    HOTSPOT_COLORS[-3],  # -3
    HOTSPOT_COLORS[-2],  # -2
    HOTSPOT_COLORS[-1],  # -1
    HOTSPOT_COLORS[0],   # 0
    HOTSPOT_COLORS[1],   # 1
    HOTSPOT_COLORS[2],   # 2
    HOTSPOT_COLORS[3],   # 3
]

# ============================================================
# COLORMAP TRANSICIONES (códigos from*10+to)
# ============================================================

TRANSITION_COLORS = {
    # Deforestación (rojos/naranjas)
    12: "#FF0000",  # Bosque→Pastizales
    13: "#FF4500",  # Bosque→Cultivos
    14: "#FF8C00",  # Bosque→Urbano
    # Urbanización (rojos oscuros)
    24: "#DC143C",  # Pastizales→Urbano
    34: "#B22222",  # Cultivos→Urbano
    # Reforestación (verdes)
    21: "#228B22",  # Pastizales→Bosque
    31: "#32CD32",  # Cultivos→Bosque
    41: "#006400",  # Urbano→Bosque
    # Cambios hídricos (azules)
    52: "#0000CD",  # Agua→Pastizales
    25: "#1E90FF",  # Pastizales→Agua
    72: "#4682B4",  # Humedales→Pastizales
    27: "#5F9EA0",  # Pastizales→Humedales
    # Sin cambio (gris claro)
    11: "#E0E0E0", 22: "#E0E0E0", 33: "#E0E0E0",
    44: "#E0E0E0", 55: "#E0E0E0", 66: "#E0E0E0", 77: "#E0E0E0",
}

TRANSITION_LABELS = {
    12: "Bosque → Pastizales", 13: "Bosque → Cultivos", 14: "Bosque → Urbano",
    21: "Pastizales → Bosque", 23: "Pastizales → Cultivos", 24: "Pastizales → Urbano", 25: "Pastizales → Agua", 27: "Pastizales → Humedales",
    31: "Cultivos → Bosque", 32: "Cultivos → Pastizales", 34: "Cultivos → Urbano",
    41: "Urbano → Bosque", 42: "Urbano → Pastizales", 43: "Urbano → Cultivos",
    52: "Agua → Pastizales", 57: "Agua → Humedales",
    72: "Humedales → Pastizales", 75: "Humedales → Agua",
}

# ============================================================
# FUNCIONES DE CREACIÓN DE COLORMAPS
# ============================================================

def get_mapbiomas_colormap() -> ListedColormap:
    """Retorna colormap ListedColormap para clases MapBiomas (0-7)."""
    return ListedColormap(MAPBIOMAS_COLOR_ARRAY, name="mapbiomas_colombia")


def get_hotspot_colormap() -> ListedColormap:
    """Retorna colormap para hotspots Gi* (-3 a 3)."""
    return ListedColormap(HOTSPOT_COLOR_ARRAY, name="hotspots_gi")


def get_transition_colormap(max_code: int = 77) -> ListedColormap:
    """
    Crea colormap para códigos de transición (0 a max_code).
    Usa colores definidos para transiciones conocidas, gris para otras.
    """
    colors = []
    for i in range(max_code + 1):
        if i in TRANSITION_COLORS:
            colors.append(TRANSITION_COLORS[i])
        elif i == 0:
            colors.append("#00000000")  # Transparente
        else:
            colors.append("#808080")  # Gris para transiciones no definidas
    return ListedColormap(colors, name="transitions")


def get_continuous_gi_colormap() -> LinearSegmentedColormap:
    """
    Colormap continuo para valores Z de Gi* (para raster interpolado).
    Azul (cold) → Gris (neutral) → Rojo (hot).
    """
    colors = [
        "#00008B",  # -3 Cold 99.9%
        "#0000FF",  # -2 Cold 99%
        "#4169E1",  # -1 Cold 95%
        "#CCCCCC",  # 0 Neutral
        "#FFD700",  # +1 Hot 95%
        "#FF8C00",  # +2 Hot 99%
        "#FF0000",  # +3 Hot 99.9%
    ]
    return LinearSegmentedColormap.from_list("gi_continuous", colors, N=256)


# ============================================================
# FUNCIONES DE LEGENDAS PARA LEAFMAP/FOLIUM
# ============================================================

def create_mapbiomas_legend_dict() -> Dict[str, str]:
    """Diccionario {nombre_clase: color_hex} para leyenda leafmap."""
    return {
        MAPBIOMAS_CLASS_NAMES[code]: MAPBIOMAS_COLORS[code]
        for code in range(1, 8)
    }


def create_hotspot_legend_dict() -> Dict[str, str]:
    """Diccionario para leyenda hotspots."""
    return {HOTSPOT_LABELS[code]: HOTSPOT_COLORS[code] for code in range(-3, 4)}


def create_transition_legend_dict(top_n: int = 10) -> Dict[str, str]:
    """Diccionario para leyenda de transiciones (top N más comunes)."""
    # Retornar las principales transiciones de interés
    main_transitions = {
        12: "Bosque → Pastizales", 13: "Bosque → Cultivos", 14: "Bosque → Urbano",
        24: "Pastizales → Urbano", 34: "Cultivos → Urbano",
        21: "Pastizales → Bosque", 31: "Cultivos → Bosque",
    }
    return {TRANSITION_LABELS[k]: TRANSITION_COLORS[k] for k in main_transitions}


# ============================================================
# FUNCIONES PARA APLICAR COLORMAP A RASTER
# ============================================================

def apply_colormap_to_array(
    array: np.ndarray,
    colormap: ListedColormap,
    vmin: int = 0,
    vmax: int = 7
) -> np.ndarray:
    """
    Aplica colormap a array de clases, retorna RGB (H, W, 3) normalizado 0-1.
    """
    # Normalizar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    normed = norm(array)
    
    # Aplicar colormap
    rgba = colormap(normed)  # (H, W, 4)
    rgb = rgba[:, :, :3]     # (H, W, 3)
    
    return rgb


def array_to_rgba_geotiff(
    array: np.ndarray,
    colormap: ListedColormap,
    output_path: str,
    transform,
    crs: str,
    nodata: int = 0
):
    """
    Guarda array clasificado como GeoTIFF RGBA con colormap embebido.
    Útil para visualización directa en QGIS.
    """
    import rasterio
    from rasterio.enums import ColorInterp
    
    # Normalizar y aplicar colormap
    rgb = apply_colormap_to_array(array, colormap)
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    
    # Crear banda alpha (0 para nodata, 255 para datos)
    alpha = np.where(array == nodata, 0, 255).astype(np.uint8)
    
    # Stack RGBA
    rgba = np.dstack([rgb_uint8, alpha])  # (H, W, 4)
    
    height, width = array.shape
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': width,
        'height': height,
        'count': 4,
        'crs': crs,
        'transform': transform,
        'nodata': 0,
        'compress': 'DEFLATE',
        'photometric': 'RGB',
        'interleave': 'pixel'
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(rgba[:, :, 0], 1)  # Red
        dst.write(rgba[:, :, 1], 2)  # Green
        dst.write(rgba[:, :, 2], 3)  # Blue
        dst.write(rgba[:, :, 3], 4)  # Alpha
        dst.colorinterp = [
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha
        ]
    
    logger.info(f"GeoTIFF RGBA guardado: {output_path}")


def create_legend_html(
    legend_dict: Dict[str, str],
    title: str = "Leyenda"
) -> str:
    """
    Crea HTML de leyenda para insertar en mapa leafmap/folium.
    """
    html = f'<div style="font-family: Arial; font-size: 12px; padding: 10px; background: white; border: 1px solid #ccc; border-radius: 4px;">'
    html += f'<b>{title}</b><br>'
    
    for label, color in legend_dict.items():
        # Convertir a hex sin alpha si tiene
        hex_color = color[:7] if color.startswith('#') and len(color) == 9 else color
        html += f'<span style="display: inline-block; width: 16px; height: 16px; background: {hex_color}; border: 1px solid #999; margin-right: 6px; vertical-align: middle;"></span> {label}<br>'
    
    html += '</div>'
    return html


def get_colormap_for_layer(layer_type: str) -> ListedColormap:
    """
    Retorna colormap apropiado según tipo de capa.
    
    layer_type: 'mapbiomas', 'hotspots', 'transitions', 'gi_continuous'
    """
    colormaps = {
        'mapbiomas': get_mapbiomas_colormap(),
        'hotspots': get_hotspot_colormap(),
        'transitions': get_transition_colormap(),
        'gi_continuous': get_continuous_gi_colormap()
    }
    return colormaps.get(layer_type, get_mapbiomas_colormap())


def get_legend_for_layer(layer_type: str) -> Dict[str, str]:
    """Retorna diccionario de leyenda para tipo de capa."""
    legends = {
        'mapbiomas': create_mapbiomas_legend_dict(),
        'hotspots': create_hotspot_legend_dict(),
        'transitions': create_transition_legend_dict()
    }
    return legends.get(layer_type, create_mapbiomas_legend_dict())


# ============================================================
# EXPORTAR COLORMAPS PARA USO EXTERNO
# ============================================================

def export_colormap_cpt(colormap: ListedColormap, output_path: str, name: str = "colormap"):
    """
    Exporta colormap a formato .cpt (GMT/Generic Mapping Tools) para QGIS.
    """
    with open(output_path, 'w') as f:
        f.write(f"# {name}.cpt\n")
        f.write("# COLOR_MODEL = RGB\n")
        n_colors = colormap.N
        for i in range(n_colors):
            rgba = colormap(i)
            r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
            f.write(f"{i} {r} {g} {b} {i+1} {r} {g} {b}\n")
        # BFN (Background, Foreground, NaN)
        f.write("B 0 0 0\n")   # Background
        f.write("F 255 255 255\n")  # Foreground
        f.write("N 0 0 0 0\n")   # NaN (transparente)
    
    logger.info(f"Colormap CPT exportado: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test colormaps
    mb_cmap = get_mapbiomas_colormap()
    hs_cmap = get_hotspot_colormap()
    trans_cmap = get_transition_colormap()
    gi_cont = get_continuous_gi_colormap()
    
    print("MapBiomas colormap:", mb_cmap.name, "N=", mb_cmap.N)
    print("Hotspot colormap:", hs_cmap.name, "N=", hs_cmap.N)
    print("Transition colormap:", trans_cmap.name, "N=", trans_cmap.N)
    print("GI continuous:", gi_cont.name)
    
    # Leyendas
    print("\nLeyenda MapBiomas:")
    for k, v in create_mapbiomas_legend_dict().items():
        print(f"  {k}: {v}")
    
    print("\nLeyenda Hotspots:")
    for k, v in create_hotspot_legend_dict().items():
        print(f"  {k}: {v}")