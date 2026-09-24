"""Proceso OFFLINE de la Tarea 2 (Fases 1 y 2 + construcción del índice de la
Fase 3). Separado de app.py (proceso ONLINE), igual que en la Tarea 1.

Uso (desde esta carpeta, tarea2_radar/):
    python build_data.py                  # solo bulk (3 meses de 2026)
    python build_data.py --incremental    # además, trae actualizaciones recientes vía API

Es idempotente: los archivos bulk no se vuelven a descargar si ya existen, y
los records recientes vía API se cachean con TTL.
"""
from __future__ import annotations

import argparse
import sys

from src.acquisition import download_all_bulk_months, fetch_recent_updates
from src.config import load_config, resolve_path
from src.embeddings import LocalEmbedder
from src.index_store import build_index
from src.ocds_parse import build_processes_table
from src.validation import validate_and_normalize, write_quality_report_markdown


def main(incremental: bool) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    cfg = load_config()
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    logs_dir = resolve_path(cfg["paths"]["logs_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=== Fase 1: adquisición (descarga masiva mensual) ===")
    bulk_paths = download_all_bulk_months(cfg)
    for p in bulk_paths:
        print(f"  - {p.name}")

    extra_records = []
    if incremental:
        print("\n=== Fase 1: actualizaciones recientes vía API ===")
        extra_records = fetch_recent_updates(cfg)
        print(f"  - {len(extra_records)} procesos con actividad reciente traídos vía API")

    print("\n=== Construcción de tabla de procesos (1 fila = 1 OCID) ===")
    df = build_processes_table(bulk_paths, extra_records=extra_records)
    print(f"  {df.attrs['build_stats']}")

    print("\n=== Fase 2: validación y normalización ===")
    df, report = validate_and_normalize(df, cfg)
    write_quality_report_markdown(report, logs_dir / "quality_report.md")
    import json
    (logs_dir / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Total procesos: {report['total_processes']}")
    print(f"  Monto ausente/cero: {report['missing_or_zero_tender_amount']['count']}")
    print(f"  Departamento inválido: {report['location_department']['invalid_after_normalization']}")

    processed_path = processed_dir / "processes.parquet"
    df.to_parquet(processed_path, index=False)
    df.to_csv(processed_dir / "processes.csv", index=False, encoding="utf-8-sig")
    print(f"\n  Guardado: {processed_path}")

    print("\n=== Fase 3: construcción del índice de embeddings (descripciones) ===")
    embedder = LocalEmbedder(cfg["embeddings"]["local"]["model_name"])
    index_dir = processed_dir / "index_local"
    manifest = build_index(df, embedder, index_dir)
    print(f"  Índice: {manifest}")

    print("\nListo. Para el dashboard: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incremental", action="store_true", help="Además del bulk, trae actualizaciones recientes vía API")
    args = parser.parse_args()
    main(incremental=args.incremental)
