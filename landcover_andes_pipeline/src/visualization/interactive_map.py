"""
Módulo Mapa Interactivo: Genera HTML standalone con leafmap
Capas: anuales, cambio, hotspots, con leyendas, popups, slider temporal
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import leafmap
import folium
from folium import plugins
import geopandas as gpd
from shapely.geometry import mapping
import json

from .colormap import (
    get_mapbiomas_colormap, get_hotspot_colormap, get_continuous_gi_colormap,
    create_mapbiomas_legend_dict, create_hotspot_legend_dict,
    create_transition_legend_dict, create_legend_html
)

logger = logging.getLogger(__name__)

# Configuración de capas base
BASEMAPS = {
    "Satélite (Esri)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "OpenStreetMap": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "CartoDB Dark": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    "CartoDB Light": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
}

# Centro aproximado región Andina Colombia
DEFAULT_CENTER = [4.5, -74.0]
DEFAULT_ZOOM = 7


def raster_to_image_overlay(
    raster_path: str,
    colormap_name: str = "mapbiomas",
    opacity: float = 0.7,
    name: str = "Capa raster"
) -> folium.raster_layers.ImageOverlay:
    """
    Convierte GeoTIFF a ImageOverlay de Folium para visualización web.
    """
    with rasterio.open(raster_path) as src:
        # Leer datos
        data = src.read(1)  # Primera banda
        bounds = src.bounds
        transform = src.transform
        crs = src.crs
        
        # Reproyectar bounds a WGS84 si es necesario
        if crs and crs.to_epsg() != 4326:
            wgs84_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
            bounds = rasterio.coords.BoundingBox(*wgs84_bounds)
        
        # Folium espera [[south, west], [north, east]]
        folium_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        
        # Obtener colormap
        from .colormap import get_colormap_for_layer
        cmap = get_colormap_for_layer(colormap_name)
        
        # Normalizar y aplicar colormap
        from .colormap import apply_colormap_to_array
        vmin, vmax = (0, 7) if colormap_name == "mapbiomas" else (-3, 3)
        norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        rgb = cmap(norm)[:, :, :3]  # (H, W, 3)
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        
        # Alpha canal (0 para nodata)
        nodata = src.nodata if src.nodata is not None else 0
        alpha = np.where(data == nodata, 0, int(opacity * 255)).astype(np.uint8)
        
        # Combinar RGBA
        rgba = np.dstack([rgb_uint8, alpha])
        
        # Crear ImageOverlay
        overlay = folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=folium_bounds,
            opacity=opacity,
            name=name,
            interactive=True,
            cross_origin=False,
            zindex=1
        )
        
        return overlay


def create_interactive_map(
    output_path: str,
    annual_rasters: Dict[int, str],
    change_raster: Optional[str] = None,
    hotspots_gpkg: Optional[str] = None,
    gi_raster: Optional[str] = None,
    aoi_gpkg: Optional[str] = None,
    config: Optional[Dict] = None,
    center: Optional[List[float]] = None,
    zoom: int = None
) -> leafmap.Map:
    """
    Crea mapa interactivo completo con leafmap.
    
    Args:
        output_path: Ruta donde guardar HTML
        annual_rasters: {año: ruta_raster} para capas anuales
        change_raster: Raster de transiciones (códigos from*10+to)
        hotspots_gpkg: GeoPackage con puntos hotspots Gi*
        gi_raster: Raster interpolado Gi* (continúo)
        aoi_gpkg: AOI como capa vectorial
        config: Configuración adicional
        center: Centro del mapa [lat, lon]
        zoom: Zoom inicial
    
    Returns:
        leafmap.Map object
    """
    if center is None:
        center = DEFAULT_CENTER
    if zoom is None:
        zoom = DEFAULT_ZOOM
    
    logger.info(f"Creando mapa interactivo en: {output_path}")
    
    # Crear mapa base con leafmap
    m = leafmap.Map(
        center=center,
        zoom=zoom,
        draw_control=False,
        measure_control=True,
        fullscreen_control=True,
        attribution_control=True
    )
    
    # Añadir capas base personalizadas
    for name, url in BASEMAPS.items():
        if "Esri" in name:
            m.add_tile_layer(
                url=url,
                name=name,
                attribution="Esri, Maxar, Earthstar Geographics",
                overlay=False,
                control=True
            )
        elif "CartoDB" in name:
            m.add_tile_layer(
                url=url,
                name=name,
                attribution="&copy; CartoDB",
                overlay=False,
                control=True
            )
    
    # Capa OSM por defecto
    m.add_basemap("OpenStreetMap")
    
    # ============================================================
    # CAPAS ANUALES DE COBERTURA
    # ============================================================
    logger.info("Añadiendo capas anuales de cobertura...")
    
    cmap_mb = get_mapbiomas_colormap()
    mb_legend = create_mapbiomas_legend_dict()
    
    for year in sorted(annual_rasters.keys()):
        raster_path = annual_rasters[year]
        if not Path(raster_path).exists():
            logger.warning(f"Raster no encontrado para {year}: {raster_path}")
            continue
        
        layer_name = f"Cobertura {year}"
        
        try:
            m.add_raster(
                raster_path,
                layer_name=layer_name,
                colormap=cmap_mb,
                opacity=0.7,
                visible=(year == max(annual_rasters.keys())),  # Solo último visible por defecto
            )
            logger.info(f"  Capa añadida: {layer_name}")
        except Exception as e:
            logger.error(f"Error añadiendo capa {year}: {e}")
    
    # ============================================================
    # CAPA DE CAMBIO 2017-2022
    # ============================================================
    if change_raster and Path(change_raster).exists():
        logger.info("Añadiendo capa de cambio...")
        try:
            trans_cmap = get_transition_colormap()  # from .colormap import ...
            trans_legend = create_transition_legend_dict()
            
            m.add_raster(
                change_raster,
                layer_name="Cambio 2017→2022",
                colormap=trans_cmap,
                opacity=0.6,
                visible=False
            )
        except Exception as e:
            logger.error(f"Error añadiendo capa cambio: {e}")
    
    # ============================================================
    # CAPA HOTSPOTS PUNTOS (Gi* significativos)
    # ============================================================
    if hotspots_gpkg and Path(hotspots_gpkg).exists():
        logger.info("Añadiendo capa hotspots (puntos)...")
        try:
            hs_gdf = gpd.read_file(hotspots_gpkg)
            
            # Filtrar solo significativos para visualización principal
            hs_sig = hs_gdf[hs_gdf["significance"] != 0].copy()
            
            if len(hs_sig) > 0:
                # Añadir como capa vectorial con estilo por significancia
                m.add_gdf(
                    hs_sig,
                    layer_name="Hotspots Deforestación (puntos)",
                    column="sig_level",
                    cmap="RdBu_r",
                    scheme="UserDefined",
                    classification_kwds={"bins": [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]},
                    legend=True,
                    legend_kwds={
                        "title": "Significancia Gi*",
                        "loc": "lower right"
                    },
                    style_kwds={
                        "fillOpacity": 0.7,
                        "radius": 4,
                        "weight": 0.5
                    },
                    visible=False
                )
                
                # Añadir popups con información
                for idx, row in hs_sig.iterrows():
                    popup_html = f"""
                    <b>Hotspot Getis-Ord Gi*</b><br>
                    <b>Significancia:</b> {row.get('significance_label', 'N/A')}<br>
                    <b>Z-score:</b> {row.get('Z_score', 0):.3f}<br>
                    <b>p-value:</b> {row.get('p_value', 0):.4f}<br>
                    <b>Gi*:</b> {row.get('Gi', 0):.3f}<br>
                    <b>Transición:</b> {row.get('transition_name', 'N/A')}<br>
                    <b>Coordenadas:</b> {row.geometry.y:.6f}, {row.geometry.x:.6f}
                    """
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=4,
                        color=get_significance_color(row.get('sig_level', 0)),
                        fill=True,
                        fillColor=get_significance_color(row.get('sig_level', 0)),
                        fillOpacity=0.7,
                        popup=folium.Popup(popup_html, max_width=300)
                    ).add_to(m)
                
                logger.info(f"  {len(hs_sig)} hotspots significativos añadidos")
        
        except Exception as e:
            logger.error(f"Error añadiendo hotspots: {e}")
    
    # ============================================================
    # CAPA GI* RASTER INTERPOLADO (SUPERFICIE CONTINUA)
    # ============================================================
    if gi_raster and Path(gi_raster).exists():
        logger.info("Añadiendo capa Gi* raster (superficie)...")
        try:
            gi_cmap = get_continuous_gi_colormap()
            
            m.add_raster(
                gi_raster,
                layer_name="Gi* Superficie (interpolado)",
                colormap=gi_cmap,
                opacity=0.5,
                visible=False
            )
        except Exception as e:
            logger.error(f"Error añadiendo Gi* raster: {e}")
    
    # ============================================================
    # CAPA AOI (LÍMITE REGIÓN ANDINA)
    # ============================================================
    if aoi_gpkg and Path(aoi_gpkg).exists():
        logger.info("Añadiendo capa AOI...")
        try:
            aoi_gdf = gpd.read_file(aoi_gpkg)
            m.add_gdf(
                aoi_gdf,
                layer_name="Región Andina (AOI)",
                style_kwds={
                    "color": "#FF0000",
                    "weight": 2,
                    "fillOpacity": 0,
                    "dashArray": "5, 5"
                },
                visible=True
            )
        except Exception as e:
            logger.error(f"Error añadiendo AOI: {e}")
    
    # ============================================================
    # CONTROLES Y LEYENDAS PERSONALIZADAS
    # ============================================================
    logger.info("Configurando controles y leyendas...")
    
    # Leyenda MapBiomas (siempre visible)
    legend_html = create_legend_html(mb_legend, "Cobertura Terrestre (MapBiomas)")
    m.add_html(legend_html, position="bottomright")
    
    # Control de capas ya incluido en leafmap
    m.add_layer_control()
    
    # Minimapa
    m.add_minimap(toggle_display=True)
    
    # Escala
    m.add_scale_bar(position="bottomleft")
    
    # Coordenadas del mouse
    m.add_coordinates(position="bottomleft")
    
    # Fullscreen
    plugins.Fullscreen(
        position="topleft",
        title="Pantalla completa",
        title_cancel="Salir",
        force_separate_button=True
    ).add_to(m)
    
    # ============================================================
    # SLIDER TEMPORAL (para capas anuales)
    # ============================================================
    # Crear TimeSliderChoropleth si hay capas anuales
    if len(annual_rasters) > 1:
        logger.info("Configurando slider temporal...")
        try:
            # leafmap no tiene TimeSliderChoropleth nativo para rasters
            # Usamos control de capas nativo + JavaScript personalizado
            add_temporal_slider_js(m, sorted(annual_rasters.keys()))
        except Exception as e:
            logger.warning(f"Slider temporal no disponible: {e}")
    
    # ============================================================
    # GUARDAR HTML
    # ============================================================
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    m.to_html(str(output_path))
    logger.info(f"Mapa interactivo guardado: {output_path}")
    
    return m


def get_significance_color(sig_level: int) -> str:
    """Color según nivel de significancia Gi*."""
    colors = {
        -3: "#00008B", -2: "#0000FF", -1: "#4169E1",
        0: "#CCCCCC",
        1: "#FFD700", 2: "#FF8C00", 3: "#FF0000"
    }
    return colors.get(sig_level, "#CCCCCC")


def add_temporal_slider_js(m: leafmap.Map, years: List[int]):
    """
    Inyecta JavaScript para slider temporal de capas anuales.
    """
    # Crear diccionario de nombres de capas
    layer_names = {f"Cobertura {y}": f"Cobertura {y}" for y in years}
    
    js_code = f"""
    <script>
    // Slider temporal para capas anuales
    document.addEventListener('DOMContentLoaded', function() {{
        var years = {years};
        var currentYearIndex = years.length - 1;
        
        // Crear UI del slider
        var sliderContainer = document.createElement('div');
        sliderContainer.style.cssText = 'position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%); z-index: 1000; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.3); font-family: Arial;';
        sliderContainer.innerHTML = `
            <div style="text-align: center;">
                <b>Año: <span id="yearDisplay">{years[currentYearIndex]}</span></b><br>
                <input type="range" id="yearSlider" min="0" max="{len(years)-1}" value="{len(years)-1}" step="1" style="width: 200px;">
            </div>
        `;
        document.body.appendChild(sliderContainer);
        
        var slider = document.getElementById('yearSlider');
        var yearDisplay = document.getElementById('yearDisplay');
        
        slider.addEventListener('input', function() {{
            var year = years[this.value];
            yearDisplay.textContent = year;
            
            // Ocultar todas las capas anuales
            years.forEach(function(y) {{
                var layerName = 'Cobertura ' + y;
                // Buscar en capas de leafmap
                if (window.leafmapLayers && window.leafmapLayers[layerName]) {{
                    window.leafmapLayers[layerName].setVisible(false);
                }}
            }});
            
            // Mostrar capa seleccionada
            var selectedLayer = 'Cobertura ' + year;
            if (window.leafmapLayers && window.leafmapLayers[selectedLayer]) {{
                window.leafmapLayers[selectedLayer].setVisible(true);
            }}
        }});
        
        // Guardar referencias a capas para acceso desde JS
        window.leafmapLayers = window.leafmapLayers || {{}};
    }});
    </script>
    """
    
    m.add_html(js_code, position="bottomleft")


def create_folium_map_alternative(
    output_path: str,
    annual_rasters: Dict[int, str],
    change_raster: Optional[str] = None,
    hotspots_gpkg: Optional[str] = None,
    aoi_gpkg: Optional[str] = None,
    center: Optional[List[float]] = None,
    zoom: int = None
) -> folium.Map:
    """
    Alternativa usando Folium directamente (más control, menos dependencias).
    """
    if center is None:
        center = DEFAULT_CENTER
    if zoom is None:
        zoom = DEFAULT_ZOOM
    
    logger.info(f"Creando mapa Folium alternativo: {output_path}")
    
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,
        control_scale=True
    )
    
    # Capas base
    for name, url in BASEMAPS.items():
        folium.TileLayer(
            tiles=url,
            name=name,
            attr="Esri" if "Esri" in name else "OSM/CartoDB",
            overlay=False,
            control=True
        ).add_to(m)
    
    # Capas anuales
    for year in sorted(annual_rasters.keys()):
        raster_path = annual_rasters[year]
        if Path(raster_path).exists():
            overlay = raster_to_image_overlay(
                raster_path,
                colormap_name="mapbiomas",
                name=f"Cobertura {year}"
            )
            overlay.add_to(m)
    
    # Capa cambio
    if change_raster and Path(change_raster).exists():
        overlay = raster_to_image_overlay(
            change_raster,
            colormap_name="transitions",
            name="Cambio 2017→2022"
        )
        overlay.add_to(m)
    
    # Hotspots
    if hotspots_gpkg and Path(hotspots_gpkg).exists():
        hs_gdf = gpd.read_file(hotspots_gpkg)
        hs_sig = hs_gdf[hs_gdf["significance"] != 0]
        
        if len(hs_sig) > 0:
            # GeoJson con estilo
            style_func = lambda feature: {
                'fillColor': get_significance_color(feature['properties'].get('sig_level', 0)),
                'color': get_significance_color(feature['properties'].get('sig_level', 0)),
                'weight': 1,
                'fillOpacity': 0.7,
                'radius': 4
            }
            
            folium.GeoJson(
                hs_sig,
                name="Hotspots Deforestación",
                style_function=style_func,
                tooltip=folium.GeoJsonTooltip(
                    fields=['significance_label', 'Z_score', 'p_value', 'transition_name'],
                    aliases=['Significancia:', 'Z-score:', 'p-value:', 'Transición:'],
                    localize=True
                ),
                popup=folium.GeoJsonPopup(
                    fields=['significance_label', 'Z_score', 'p_value', 'Gi', 'transition_name', 'x', 'y'],
                    aliases=['Significancia:', 'Z-score:', 'p-value:', 'Gi*:', 'Transición:', 'Lon:', 'Lat:'],
                    localize=True
                )
            ).add_to(m)
    
    # AOI
    if aoi_gpkg and Path(aoi_gpkg).exists():
        aoi_gdf = gpd.read_file(aoi_gpkg)
        folium.GeoJson(
            aoi_gdf,
            name="Región Andina (AOI)",
            style_function=lambda x: {
                'color': '#FF0000',
                'weight': 2,
                'fillOpacity': 0,
                'dashArray': '5, 5'
            }
        ).add_to(m)
    
    # Leyenda MapBiomas
    legend_html = create_legend_html(create_mapbiomas_legend_dict(), "Cobertura Terrestre (MapBiomas)")
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Controles
    folium.LayerControl(collapsed=False).add_to(m)
    plugins.MiniMap(toggle_display=True).add_to(m)
    plugins.Fullscreen().add_to(m)
    
    # Guardar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    
    logger.info(f"Mapa Folium guardado: {output_path}")
    return m


def create_simple_preview_map(
    raster_path: str,
    output_html: str,
    title: str = "Vista Previa"
) -> str:
    """
    Crea mapa HTML simple para vista rápida de un solo raster.
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        if src.crs and src.crs.to_epsg() != 4326:
            wgs84_bounds = transform_bounds(src.crs, "EPSG:4326", *bounds)
            bounds = rasterio.coords.BoundingBox(*wgs84_bounds)
        
        center = [(bounds.top + bounds.bottom) / 2, (bounds.left + bounds.right) / 2]
    
    m = folium.Map(location=center, zoom_start=8)
    
    # Añadir raster
    overlay = raster_to_image_overlay(raster_path)
    overlay.add_to(m)
    
    # Leyenda
    legend = create_legend_html(create_mapbiomas_legend_dict(), "Cobertura")
    m.get_root().html.add_child(folium.Element(legend))
    
    folium.LayerControl().add_to(m)
    
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    
    return str(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Módulo interactive_map listo")