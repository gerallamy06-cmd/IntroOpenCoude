# Pipeline Cobertura Terrestre - Región Andina Colombia

🛰️ **Detección automática de cobertura terrestre para la región Andina de Colombia** usando modelos geoespaciales de código abierto y datos satelitales de Planetary Computer.

## 📋 Características Principales

| Característica | Detalle |
|----------------|---------|
| **Fuente de datos** | Planetary Computer - Esri 10m Land Cover (IO-LULC Annual v02) |
| **Período** | 2017-2022 (6 años, serie temporal anual) |
| **Resolución** | 10 metros |
| **Clases** | MapBiomas Colombia simplificado (7 clases) |
| **Región** | 14 departamentos andinos (auto-descargado desde GADM v4.1) |
| **Análisis temporal** | Matrices de transición anuales + global 2017→2022 |
| **Hotspots** | Getis-Ord Gi* para clusters de deforestación/urbanización |
| **Salidas** | GeoTIFF COG, GPKG, CSV, HTML interactivo |
| **API Key** | No requerida (acceso público con retry exponencial) |

## 🗺️ Clases MapBiomas Colombia (7)

| Código | Clase | Color |
|--------|-------|-------|
| 1 | **Bosque** | `#129912` |
| 2 | **Pastizales** | `#FFFFB2` |
| 3 | **Cultivos** | `#E8C16F` |
| 4 | **Urbano** | `#E86B4C` |
| 5 | **Agua** | `#4C9CE8` |
| 6 | **Suelo desnudo** | `#D68B4C` |
| 7 | **Humedales** | `#A6D8F0` |

> **Mapeo desde IO-LULC:** Water(1)→Agua(5), Trees(2)→Bosque(1), Flooded Veg(4)→Humedales(7), Crops(5)→Cultivos(3), Built(7)→Urbano(4), Bare(8)→Suelo desnudo(6), Rangeland(11)→Pastizales(2)

## 🚀 Instalación Rápida

### Opción A: Conda (Recomendado para dependencias binarias)
```bash
# Crear entorno
conda create -n landcover_andes python=3.11
conda activate landcover_andes

# Dependencias geoespaciales (GDAL, libpysal, esda)
conda install -c conda-forge gdal libpysal esda rasterio geopandas

# Dependencias Python
pip install -r requirements.txt
```

### Opción B: Solo pip (puede requerir GDAL previo)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## ⚙️ Configuración

Edita `config.yaml` para personalizar:

```yaml
output:
  base_dir: "C:/Users/TuUsuario/Documentos/landcover_output/"  # Cambia esta ruta

planetary_computer:
  api_key: ""  # Opcional: mejora rate limits
```

## ▶️ Ejecución

### Pipeline completo (30-60 min)
```bash
python run_pipeline.py
```

### Con opciones avanzadas
```bash
python main.py -c config.yaml -v  # Verbose
python main.py --step all         # Pipeline completo
```

## 📁 Estructura de Salidas

```
landcover_output/
├── rasters/
│   ├── landcover_2017.tif ... landcover_2022.tif     # Cobertura anual
│   ├── landcover_latest.tif                           # Último año (2022)
│   ├── transition_2017_2018.tif ... transition_2021_2022.tif  # Cambio anual
│   ├── transition_2017_2022.tif                       # Cambio global
│   └── hotspots_gistar.tif                            # Gi* interpolado (Z-scores)
├── vectors/
│   ├── transitions_2017_2018.gpkg ... transitions_2021_2022.gpkg
│   ├── transitions_2017_2022.gpkg
│   ├── hotspots_deforestation.gpkg                    # Hot spots (deforestación)
│   ├── hotspots_regeneration.gpkg                     # Cold spots (regeneración)
│   └── aoi_andina.gpkg                                # Límite región andina
├── stats/
│   ├── area_by_year.csv                               # Área por clase y año
│   ├── area_by_class.csv                              # Promedio temporal por clase
│   ├── transition_matrix_2017_2018.csv ...            # Matrices anuales
│   ├── transition_matrix_global.csv                   # Matriz 2017→2022
│   ├── net_change_global.csv                          # Cambio neto (ha)
│   ├── dominant_transitions_global.csv                # Top transiciones
│   ├── annual_change_rates.csv                        # Tasas %/año
│   ├── hotspots_summary.csv                           # Resumen Gi*
│   ├── hotspots_by_transition.csv                     # Hotspots por tipo
│   └── consolidated_report.json                       # Reporte completo
├── maps/
│   └── andes_landcover_2017_2022_hotspots.html        # 🌐 MAPA INTERACTIVO
└── logs/
    └── pipeline_YYYYMMDD_HHMM.log
```

## 🌐 Mapa Interactivo (HTML)

El archivo `andes_landcover_2017_2022_hotspots.html` incluye:

- **Capas base:** Satélite (Esri), OpenStreetMap, CartoDB Dark/Light
- **Cobertura anual 2017-2022** (checkbox + slider temporal)
- **Cambio 2017→2022** (códigos transición, opacidad adjustable)
- **Hotspots Gi*** (puntos: Hot/Cold/No significativo + popups informativos)
- **Superficie Gi* interpolada** (raster continuo Z-scores)
- **AOI Región Andina** (límite punteado rojo)
- **Leyenda dinámica** (cambia según capa activa)
- **Controles:** Mini-mapa, escala, coordenadas, pantalla completa

## 🔬 Análisis Getis-Ord Gi* (Hotspots)

Detecta clusters estadísticamente significativos (p < 0.05) de:

| Transición objetivo | Interpretación |
|---------------------|----------------|
| Bosque → Pastizales (12) | Deforestación para ganadería |
| Bosque → Cultivos (13) | Deforestación para agricultura |
| Bosque → Urbano (14) | Deforestación para expansión urbana |
| Pastizales → Urbano (24) | Urbanización de áreas rurales |

**Salida:** Puntos con Z-score, p-value simulado (999 permutaciones), clasificación de significancia (95%/99%/99.9%).

## 📊 Ejemplo de Resultados

### Cambio Neto Global (ha)
| Clase | Ganancia | Pérdida | Neto |
|-------|----------|---------|------|
| Bosque | +12,450 | -89,320 | **-76,870** |
| Pastizales | +45,210 | -12,340 | **+32,870** |
| Cultivos | +28,900 | -5,670 | **+23,230** |
| Urbano | +8,750 | -120 | **+8,630** |

### Transiciones Dominantes 2017→2022
1. **Bosque → Pastizales**: 42,150 ha (54.8% del cambio)
2. **Bosque → Cultivos**: 18,920 ha (24.6%)
3. **Pastizales → Urbano**: 6,840 ha (8.9%)

## 🏗️ Arquitectura del Pipeline

```
main.py (Orquestador)
├── src/data_access/
│   ├── aoi.py                    # GADM download + filtro 14 deptos andinos
│   └── planetary_computer.py     # STAC search + stackstac + retry/backoff
├── src/processing/
│   ├── mosaic_clip.py            # Mosaico tiles + recorte AOI exacto
│   ├── remap_classes.py          # IO-LULC 9 → MapBiomas 7 (vectorizado)
│   ├── change_detection.py       # 7 matrices (6 anuales + 1 global)
│   ├── hotspots.py               # Getis-Ord Gi* (libpysal/esda)
│   └── stats.py                  # Estadísticas área, tendencias, resúmenes
├── src/visualization/
│   ├── colormap.py               # Colormaps MapBiomas + Gi* oficiales
│   └── interactive_map.py        # leafmap → HTML standalone
└── src/export/
    ├── raster_export.py          # COG GeoTIFFs + metadatos
    └── vector_export.py          # GPKG transiciones + hotspots
```

## 🔧 Requisitos del Sistema

- **Python**: 3.10+
- **RAM**: ≥16 GB recomendado (chunking 256×256 para datasets grandes)
- **Disco**: ≥10 GB libres (rasters temporales + outputs)
- **Red**: Conexión estable a Planetary Computer (Azure) y GADM (UC Davis)
- **GDAL**: 3.4+ (para COG, VRT, reproyección)

## 📚 Referencias y Fuentes de Datos

1. **Planetary Computer STAC API**: https://planetarycomputer.microsoft.com/api/stac/v1
2. **Esri 10m Land Cover (IO-LULC Annual v02)**: 2017-2022, 10m, 9 clases
3. **GADM v4.1 Colombia**: Límites administrativos (14 departamentos andinos)
4. **MapBiomas Colombia**: Esquema de clasificación 7 clases
5. **Getis-Ord Gi***: Análisis de clusters espaciales (libpysal/esda)

## 🤝 Créditos

- **Desarrollado para**: IA Aplicada al Territorio - Onceavo Semestre
- **Tecnologías**: Python, Planetary Computer, stackstac, leafmap, libpysal
- **Inspiración**: MapBiomas Colombia, Microsoft AI for Earth, NASA IMPACT

## ⚠️ Limitaciones Conocidas

1. **Sin API key**: Rate limiting en Planetary Computer (retry automático configurado)
2. **Memoria**: Procesa por chunks; AOI muy grande puede requerir tileado manual
3. **Clases**: No distingue bosque denso/secundario (requeriría Hansen GFC o LiDAR)
4. **Años**: Limitado a 2017-2022 (disponibilidad IO-LULC Annual v02)
5. **Gi* performance**: >100k puntos usa submuestreo sistemático

## 📝 Licencia

MIT License - Uso libre para investigación y educación.

---

**¿Problemas?** Revisa `logs/pipeline_*.log` para detalles de ejecución.