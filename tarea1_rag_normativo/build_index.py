"""Proceso OFFLINE de indexación (Fases 1 y 2).

Separado del proceso ONLINE de consulta (app.py / src/rag_engine.py) según la
arquitectura pedida: este script no expone ninguna interfaz de usuario, solo
construye/actualiza el índice a partir de los PDF fuente.

Uso (desde esta carpeta, tarea1_rag_normativo/):
    python build_index.py

Es idempotente: se puede volver a correr sin duplicar fragmentos ni recomputar
embeddings ya calculados (ver src/index_store.py).
"""
from __future__ import annotations

import json
import sys

from src.run_phase1 import run_phase1
from src.run_phase2 import run_phase2


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    print("=== Fase 1: verificación de fuentes, extracción y limpieza ===")
    phase1_summary = run_phase1()
    for doc in phase1_summary["documents"]:
        print(f"  - {doc['doc_id']}: {doc['page_count']} páginas, "
              f"extraíble={doc['is_text_extractable']}, "
              f"páginas sospechosas={doc['suspicious_pages']}")

    print("\n=== Fase 2: chunking, embeddings locales e índice ===")
    phase2_report = run_phase2()
    for c in phase2_report["configs"]:
        print(f"  - {c['config_name']}: {c['total_fragments']} fragmentos "
              f"(long. media={c['length_chars_mean']} chars), "
              f"IDs únicos={c['ids_are_unique']}")
    print(f"  Configuración seleccionada para el motor RAG: {phase2_report['selected_config']}")

    print("\nÍndice listo en data/processed/. Para la comparación con embeddings de "
          "OpenAI y el cálculo de Recall@k, corre: python -m eval.run_eval")


if __name__ == "__main__":
    main()
