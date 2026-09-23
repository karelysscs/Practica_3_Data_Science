"""Punto de entrada de la Fase 4 (evaluación y comparación de embeddings).

Requiere que build_index.py ya se haya corrido (índice local construido).
Si OPENAI_API_KEY está configurada en .env, también construye y evalúa el
índice con text-embedding-3-small; si no, reporta esa fila como omitida.

Uso (desde tarea1_rag_normativo/):
    python -m eval.run_eval
"""
from __future__ import annotations

import sys

sys.path.insert(0, "..") if __package__ is None else None

from src.run_phase4_eval import run_phase4_eval

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    result = run_phase4_eval()
    print(f"Resultados guardados en eval/results_embeddings_comparison.json "
          f"y logs/phase4_evaluation_report.md")
    for r in result["embedders"]:
        if r.get("skipped"):
            print(f"- {r['embedder']}: OMITIDO ({r['reason']})")
        else:
            print(f"- {r['embedder']}: Recall@k={r['recall_at_k']}")
