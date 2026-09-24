"""Punto de entrada de la Fase 4. Uso (desde tarea2_radar/): python -m eval.run_eval"""
from __future__ import annotations

import sys

from src.run_eval import run_eval

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    res = run_eval()
    print(f"Recall@k: {res['recall_at_k']}")
    print(f"Calibración de umbral: {res['threshold_calibration']}")
