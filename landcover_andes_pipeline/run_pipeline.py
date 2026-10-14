#!/usr/bin/env python3
"""
Script de ejecución simple para el pipeline.
Uso: python run_pipeline.py
"""
import sys
import os

# Añadir directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import run_pipeline, setup_logging

if __name__ == "__main__":
    print("=" * 60)
    print("PIPELINE COBERTURA TERRESTRE - REGIÓN ANDINA COLOMBIA")
    print("Fuente: Planetary Computer (Esri 10m IO-LULC Annual v02)")
    print("Clases: MapBiomas Colombia (7 clases)")
    print("Análisis: Cambio 2017-2022 + Hotspots Getis-Ord Gi*")
    print("=" * 60)
    
    setup_logging()
    
    try:
        results = run_pipeline("config.yaml")
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETADO")
        print(f"📁 Outputs en: {results['output_dirs']['stats']}")
        print(f"🌐 Mapa interactivo: {results['maps'].get('interactive_html', 'N/A')}")
        print(f"⏱️  Duración: {results.get('duration_formatted', 'N/A')}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrumpido por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)