# Bitácora de Planificación: Pipeline Cobertura Terrestre Región Andina Colombia

**Fecha de inicio:** 2026-10-07  
**Proyecto:** landcover_andes_pipeline  
**Ubicación:** `C:\Users\Acer\Documents\UD\ONCEAVO SEMESTRE\IA APLICADA AL TERRITORIO\landcover_andes_pipeline\`  
**Objetivo:** Crear un programa en Python para detectar el tipo de cobertura terrestre de una zona de Colombia, empleando modelos de inteligencia artificial geoespacial de código abierto.

---

## 1. SOLICITUD INICIAL

> "Crear un programan en python que me permita detectar el tipo de cobertura terrestre de una zona de Colombia, empleando modelos de inteligencia artificial geoespacial de código abierto"

---

## 2. FASE DE INVESTIGACIÓN (Web Searches Realizadas)

### 2.1 Búsqueda 1: Modelos geoespaciales open source para Colombia
**Query:** `open source geospatial AI models land cover classification Colombia 2024`  
**Resultados clave:**
- MapBiomas Colombia Collection 3 (GEE + Random Forest, 1985-2025, 7-8 clases)
- Microsoft Landcover Orinoquía (U-Net + Landsat 8, 12 clases, archivado)
- esteban-russi/sentinel2-dl-deforestation (U-Net + Dynamic World, Colombia)
- Felipe-RA/udeai_forest (VGGUdeA + Sentinel-2, Antioquia)
- CoL-Seg dataset (Colombian Diverse Landscapes Semantic Segmentation)
- TorchGeo, dl4eo, segment-geospatial, geoai, unbihexium librerías

### 2.2 Búsqueda 2: Librerías Python geoespaciales deep learning
**Query:** `Python libraries geospatial deep learning land cover segmentation satellite imagery`  
**Resultados clave:**
- TorchGeo (PyTorch domain library, datasets, samplers, pre-trained models)
- segment-geospatial (SAM para datos geoespaciales)
- geoai (framework integral PyTorch + Transformers + TorchGeo)
- dl4eo (pipeline Sentinel-2 + Sentinel-1 + DEM → segmentación)
- SemanticSeg4EO (15+ arquitecturas: SegFormer, ConvNeXt, HRNet, Swin-UNet)

### 2.3 Búsqueda 3: Modelos pre-entrenados land cover Colombia
**Query:** `pre-trained land cover models Colombia Sentinel-2 open source semantic segmentation`  
**Resultados clave:**
- Esri 10m Land Cover (Planetary Computer, 9 clases, 2017-2022, 10m)
- Dynamic World (Google Earth Engine, 9 clases, near real-time)
- ESA WorldCover (11 clases, 2020/2021, 10m, COGs en AWS S3)
- Prithvi Foundation Model (IBM-NASA, 100M/300M/600M params, fine-tuning)
- FLAIR-HUB models (Hugging Face, UTAE encoder, 15 clases LC)

### 2.4 Búsqueda 4: Fine-tuning Prithvi para land cover
**Query:** `Prithvi foundation model land cover Colombia fine-tuning Sentinel-2`  
**Resultados clave:**
- Paper: Fine-tuning Prithvi para LULC (mIoU 62.37% vs U-Net 46.8%)
- NASA-IMPACT/hls-foundation-os (ejemplos fine-tuning flood, burn scars, crops)
- easare377/Prithvi-EO-Segmentation (TerraTorch + Prithvi-EO 1.0/2.0)
- Prithvi-EO-2.0-600M-TL: mejor modelo con embeddings temporales+locación

### 2.5 Búsqueda 5: Planetary Computer + Microsoft ejemplos
**Query:** `Microsoft Planetary Computer land cover Colombia Sentinel-2 semantic segmentation Python`  
**Resultados clave:**
- PlanetaryComputerExamples: io-lulc-example.ipynb, esa-worldcover-example.ipynb
- Tutorial landcover.ipynb (NAIP + U-Net + Dask GPU)
- Acceso STAC API + stackstac + odc-stac para lazy loading
- Datasets: io-lulc (2017-2022), io-lulc-annual-v02 (9 clases), esa-worldcover (2020/2021)

---

## 3. FASE DE CLARIFICACIÓN (Preguntas al Usuario)

### Pregunta 1: Enfoque del programa
**Opciones presentadas:**
1. Solo inferencia (recomendado para empezar) ✓ **SELECCIONADO**
2. Fine-tuning de modelos fundacionales
3. Entrenamiento completo personalizado

### Pregunta 2: Esquema de clases
**Opciones presentadas:**
1. Clases MapBiomas Colombia (7-8 clases) ✓ **SELECCIONADO**
2. Clases Esri 10m / Dynamic World (9-10 clases)
3. Clases ESA WorldCover (11 clases)

### Pregunta 3: Zona geográfica
**Opciones presentadas:**
1. Colombia completa
2. Región/bioma específico ✓ **SELECCIONADO** → **Andina**
3. Área de interés personalizada

### Pregunta 4: Formato de salida
**Opciones presentadas (múltiple):**
- Mapa interactivo ✓ **SELECCIONADO**
- (Otras: Raster GeoTIFF, Vector, Estadísticas - se añadieron después)

### Pregunta 5: Bioma específico
**Opciones:**
- Orinoquía, Amazonía, **Andina ✓ SELECCIONADO**, Caribe, Pacífica, Insular

### Pregunta 6: Shapefile
**Respuesta:** "Búscalo automáticamente" → Se usará GADM v4.1 nivel 1 filtrando 14 departamentos andinos

### Pregunta 7: Análisis temporal
**Respuesta:** "Serie completa 2017-2022 (6 años)" → 6 años de datos IO-LULC Annual v02

### Pregunta 8: Directorio de salida
**Respuesta:** "Ruta personalizada" → Se configurará en config.yaml

### Pregunta 9: Confirmaciones finales
- ✅ Descargar shapefile automáticamente (GADM)
- ✅ Sin API key Planetary Computer (acceso público con retry)
- ✅ Matrices de transición anuales consecutivas + global 2017→2022
- ✅ Análisis Getis-Ord Gi* para hotspots de deforestación/urbanización

---

## 4. DECISIONES TÉCNICAS CLAVE

| Aspecto | Decisión | Justificación |
|---------|----------|---------------|
| **Fuente de datos** | Planetary Computer - Esri 10m IO-LULC Annual v02 | Serie temporal 2017-2022, 10m, 9 clases, acceso STAC público, cloud-free composites |
| **Clases objetivo** | MapBiomas Colombia simplificado (7 clases) | Usuario lo solicitó; remapeo directo desde IO-LULC 9 clases |
| **AOI** | GADM v4.1 Colombia admin level 1 → 14 departamentos andinos | Fuente oficial, acceso programático, geometrías validadas |
| **Arquitectura** | Pipeline modular (data_access → processing → visualization → export) | Mantenibilidad, reutilización, testing independiente |
| **Procesamiento** | stackstac + odc-stac (lazy loading, chunking 256×256) | Memoria eficiente, datasets grandes (6 años × ~15 tiles) |
| **Rate limiting** | Retry exponencial (max 5, backoff 2x) sin API key | Planetary Computer público tiene límites; retry maneja 429/5xx |
| **Cambio temporal** | 6 matrices anuales consecutivas + 1 global 2017→2022 | Análisis de tendencias año a año + cambio neto período completo |
| **Hotspots** | Getis-Ord Gi* (libpysal/esda) con 999 permutaciones | Estadístico espacial estándar, p-values simulados, Queen contiguity |
| **Transiciones objetivo Gi*** | Bosque→Pastizales, Bosque→Cultivos, Bosque→Urbano, Pastizales→Urbano | Principales drivers de cambio en Andes colombianos |
| **Salida raster** | Cloud Optimized GeoTIFF (COG) con DEFLATE | Streaming eficiente, nube-ready, compresión buena |
| **Salida vector** | GeoPackage (GPKG) | Formato moderno, un archivo, soporta múltiples capas |
| **Mapa web** | leafmap → HTML standalone | Sin servidor, interactivo, capas múltiples, leyenda dinámica |
| **CRS** | EPSG:4326 (procesamiento) → EPSG:3116 (cálculo área ha) | STAC nativo en 4326; área precisa en proyección Colombia |

---

## 5. MAPEO DE CLASES: IO-LULC (9) → MAPBIOMAS COLOMBIA (7)

| IO-LULC Value | IO-LULC Clase | MapBiomas Value | MapBiomas Clase |
|---------------|---------------|-----------------|-----------------|
| 1 | Water | 5 | Agua |
| 2 | Trees | 1 | Bosque |
| 4 | Flooded Vegetation | 7 | Humedales |
| 5 | Crops | 3 | Cultivos |
| 7 | Built Area | 4 | Urbano |
| 8 | Bare Ground | 6 | Suelo desnudo |
| 11 | Rangeland | 2 | Pastizales |
| 9 | Snow/Ice | 0 | NoData |
| 10 | Clouds | 0 | NoData |

> **Nota:** No se distingue Bosque denso vs. Secundario (usuario lo confirmó). Para esa distinción se requeriría Hansen GFC, LiDAR o umbral NDVI/canopy cover.

---

## 6. DEPARTAMENTOS ANDINOS (GADM Nivel 1 - 14)

1. Antioquia
2. Boyacá
3. Caldas
4. Cauca
5. Cesar
6. Cundinamarca
7. Huila
8. Nariño
9. Norte de Santander
10. Quindío
11. Risaralda
12. Santander
13. Tolima
14. Valle del Cauca

**Fuente:** GADM v4.1 (gadm.org) - Administrative boundaries
**CRS:** EPSG:4326 (WGS84)
**Descarga:** Automática via `pooch` con cache local

---

## 7. ESTRUCTURA FINAL DEL PROYECTO

```
landcover_andes_pipeline/
├── config.yaml                    # Configuración central (YAML)
├── main.py                        # Orquestador principal (CLI)
├── run_pipeline.py                # Entry point simple
├── requirements.txt               # Dependencias
├── README.md                      # Documentación
├── BITACORA_PLANIFICACION.md      # Este archivo
└── src/
    ├── __init__.py
    ├── data_access/
    │   ├── __init__.py
    │   ├── aoi.py                 # GADM download + filtro 14 deptos
    │   └── planetary_computer.py  # STAC search + stackstac + retry
    ├── processing/
    │   ├── __init__.py
    │   ├── mosaic_clip.py         # Mosaico + recorte AOI
    │   ├── remap_classes.py       # IO-LULC 9 → MapBiomas 7
    │   ├── change_detection.py    # 7 matrices (6 anuales + 1 global)
    │   ├── hotspots.py            # Getis-Ord Gi* (libpysal/esda)
    │   └── stats.py               # DataFrames áreas + transiciones
    ├── visualization/
    │   ├── __init__.py
    │   ├── colormap.py            # Colores MapBiomas + hotspots
    │   └── interactive_map.py     # leafmap HTML completo
    └── export/
        ├── __init__.py
        ├── raster_export.py       # COG GeoTIFFs + metadatos
        └── vector_export.py       # GPKG transiciones + hotspots
```

---

## 8. DEPENDENCIAS (requirements.txt)

```txt
# Core geospatial
rasterio>=1.3
geopandas>=0.14
shapely>=2.0
pyproj>=3.5

# Planetary Computer / STAC
planetary-computer>=0.4
pystac-client>=0.7
stackstac>=0.3
odc-stac>=0.3

# Data processing
numpy>=1.24
pandas>=2.0
tqdm>=4.66

# Spatial stats (Getis-Ord Gi*)
libpysal>=4.8
esda>=2.6
scipy>=1.11

# GADM download
requests>=2.31
pooch>=1.7

# Visualization
leafmap>=0.30
folium>=0.15
matplotlib>=3.7

# Config & utils
pyyaml>=6.0
```

**Recomendación instalación:**
```bash
conda create -n landcover_andes python=3.11
conda activate landcover_andes
conda install -c conda-forge gdal libpysal esda rasterio geopandas
pip install -r requirements.txt
```

---

## 9. SALIDAS ESPERADAS

### Rasters (COG GeoTIFF)
```
landcover_output/rasters/
├── landcover_2017.tif ... landcover_2022.tif          # 6 años
├── landcover_latest.tif                               # Alias 2022
├── transition_2017_2018.tif ... transition_2021_2022.tif  # 6 anuales
├── transition_2017_2022.tif                           # Global
└── hotspots_gistar.tif                                # Superficie Gi*
```

### Vectores (GPKG)
```
landcover_output/vectors/
├── transitions_2017_2018.gpkg ... transitions_2021_2022.gpkg
├── transitions_2017_2022.gpkg
└── hotspots_deforestation.gpkg
```

### Estadísticas (CSV)
```
landcover_output/stats/
├── area_by_year_class.csv              # Año, clase, ha, %
├── transition_matrix_2017_2018.csv ... transition_matrix_2021_2022.csv
├── transition_matrix_2017_2022.csv     # 7×7 matriz global
├── change_summary_annual.csv           # Resumen por año
├── change_summary_global.csv           # Resumen 2017→2022
└── hotspots_summary.csv                # Gi* stats
```

### Mapa Web (HTML)
```
landcover_output/maps/
└── andes_landcover_2017_2022_hotspots.html
```

**Capas del mapa:**
- 6 capas anuales cobertura (checkbox + slider)
- 1 capa cambio 2017→2022 (opacidad adjustable)
- 1 capa hotspots puntos (Hot/Cold/No sig.)
- 1 capa hotspots raster (superficie Gi*)
- Leyenda dinámica, popup informativo, mini-mapa, escala

---

## 10. FLUJO DE EJECUCIÓN (main.py)

```python
def run_pipeline(config_path: str = "config.yaml"):
    # 1. Cargar configuración
    config = load_config(config_path)
    
    # 2. AOI: Descargar/filtar GADM → GeoDataFrame 14 deptos
    aoi_gdf = get_andean_aoi(config)
    
    # 3. Data Access: STAC search + stackstac load (6 años)
    stack = download_lulc_stack(config, aoi_gdf)
    
    # 4. Procesamiento: Mosaico + Recorte AOI exacto
    clipped_stack = mosaic_and_clip(stack, aoi_gdf)
    
    # 5. Remapeo: IO-LULC 9 → MapBiomas 7
    remapped_stack = remap_classes(clipped_stack, config)
    
    # 6. Export rasters anuales
    export_annual_rasters(remapped_stack, config)
    
    # 7. Change Detection: Matrices + rasters transición
    change_results = detect_changes(remapped_stack, config)
    export_transition_rasters(change_results, config)
    export_transition_vectors(change_results, config)
    export_stats(change_results, config)
    
    # 8. Hotspots Getis-Ord Gi*
    if config.hotspots.enabled:
        hotspots_results = compute_hotspots(change_results, config)
        export_hotspots(hotspots_results, config)
    
    # 9. Mapa Interactivo
    create_interactive_map(config, hotspots_results if enabled else None)
    
    # 10. Logging final
    log_summary(config, results)
```

---

## 11. CONSIDERACIONES DE RENDIMIENTO Y MEMORIA

| Componente | Estrategia |
|------------|------------|
| **Descarga tiles** | Paralelismo con `stackstac.load(chunks=256)` + dask |
| **Mosaico** | Lazy evaluation (xarray/dask) → compute solo al recortar |
| **Recorte AOI** | `rasterio.mask.mask(crop=True)` por tile → merge final |
| **Remapeo** | `numpy.select()` vectorizado (evita loops Python) |
| **Matrices transición** | `sklearn.metrics.confusion_matrix` o `numpy.bincount` 2D |
| **Getis-Ord Gi*** | Muestreo sistemático si >100k puntos; Queen weights sparse |
| **Export COG** | `driver="COG", compress="DEFLATE", predictor=2` |
| **Logging** | Rotating file handler + console (tqdm para progreso) |

---

## 12. RIESGOS Y MITIGACIONES

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Rate limiting PC sin API key | Alta | Medio | Retry exponencial (5 reintentos, backoff 2x), delays aleatorios |
| Memoria insuficiente (stack 6 años) | Media | Alto | Chunking 256×256; procesar por tiles si >2GB RAM |
| GADM download falla | Baja | Medio | Cache local (pooch); fallback URL directa; shapefile manual opcional |
| libpysal/esda instalación | Media | Alto | `conda install -c conda-forge` obligatorio en README |
| Gi* muy lento (>100k puntos) | Media | Medio | Muestreo sistemático (1 de cada N); weights sparse |
| Tiles faltantes en algún año | Baja | Medio | Validar cobertura STAC; log warnings; continuar con disponibles |

---

## 13. TIEMPO ESTIMADO DE EJECUCIÓN

| Fase | Tiempo (sin API key) |
|------|---------------------|
| Setup + config + AOI download | 2-5 min |
| STAC search + tiles download (6 años × ~15 tiles) | 15-30 min |
| Mosaico + recorte + remapeo | 5-10 min |
| Matrices transición (7) + estadísticas | 2-5 min |
| Getis-Ord Gi* (999 permutaciones) | 5-15 min |
| Export rasters (13 COGs) + vectores (7 GPKG) | 5-10 min |
| Mapa HTML interactivo | 2-5 min |
| **TOTAL ESTIMADO** | **30-60 min** |

---

## 14. VALIDACIÓN PRE-EJECUCIÓN (Checklist)

- [ ] `config.yaml` existe con `output.base_dir` válido y escribible
- [ ] Python 3.10+ con GDAL/rasterio funcional (`rasterio --version`)
- [ ] `conda install -c conda-forge gdal libpysal esda` ejecutado
- [ ] `pip install -r requirements.txt` sin errores
- [ ] Espacio en disco: ≥10 GB libres
- [ ] Conexión internet estable (test: `curl https://planetarycomputer.microsoft.com/api/stac/v1`)
- [ ] Permisos escritura en `C:/Users/Acer/Documentos/landcover_output/`

---

## 15. PRÓXIMOS PASOS (Implementación)

1. **Crear estructura de directorios** ✓ (hecho)
2. **Escribir config.yaml** ✓ (hecho)
3. **Implementar módulos src/** (13 archivos)
4. **Crear main.py, run_pipeline.py, requirements.txt, README.md**
5. **Testing: ejecutar pipeline completo**
6. **Validar salidas: rasters, vectores, CSV, HTML**
7. **Documentar lecciones aprendidas en bitácora**

---

## 16. DECISIONES PENDIENTES / ABIERTAS

*Ninguna - todas las decisiones clave fueron confirmadas por el usuario.*

---

## 17. REFERENCIAS CLAVE

1. **Planetary Computer STAC API:** https://planetarycomputer.microsoft.com/api/stac/v1
2. **Esri 10m Land Cover (IO-LULC):** https://planetarycomputer.microsoft.com/dataset/io-lulc-annual-v02
3. **GADM v4.1 Colombia:** https://gadm.org/download_country.html
4. **MapBiomas Colombia Collection 3:** https://github.com/mapbiomas/colombia-collection-3
5. **Getis-Ord Gi* (libpysal):** https://pysal.org/libpysal/generated/libpysal.weights.Queen.html
6. **stackstac documentation:** https://stackstac.readthedocs.io/
7. **leafmap documentation:** https://leafmap.org/
8. **COG specification:** https://cogeo.org/

---

**Fin de la bitácora de planificación.**  
**Próxima fase:** Implementación completa del código según esta especificación.