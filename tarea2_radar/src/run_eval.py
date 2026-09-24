"""Fase 4 — Evaluación del RAG híbrido: Recall@k sobre 13 preguntas reales
(10 in-scope con procesos relevantes conocidos + 3 out-of-scope para calibrar
el umbral de abstención, reutilizando y recalibrando la lógica de la Tarea 1).

No importa UI. Uso: python -m src.run_eval   (desde tarea2_radar/)
"""
from __future__ import annotations

import json
import statistics
import sys
import time

import pandas as pd

from src.config import load_config, resolve_path
from src.embeddings import LocalEmbedder, cosine_search
from src.hybrid_rag import Filters, apply_filters
from src.index_store import load_index


def run_eval() -> dict:
    cfg = load_config()
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    eval_dir = resolve_path(cfg["paths"]["eval_dir"])
    logs_dir = resolve_path(cfg["paths"]["logs_dir"])

    df = pd.read_parquet(processed_dir / "processes.parquet")
    vectors, manifest = load_index(processed_dir / "index_local")
    embedder = LocalEmbedder(cfg["embeddings"]["local"]["model_name"])

    eval_set = json.loads((eval_dir / "eval_set.json").read_text(encoding="utf-8"))
    questions = eval_set["questions"]
    k_values = cfg["evaluation"]["k_values"]
    max_k = max(k_values)

    per_question = []
    for q in questions:
        filters = Filters(**q.get("filters", {}))
        filtered = apply_filters(df, filters)
        filtered_vectors = vectors[filtered.index.to_numpy()] if len(filtered) else vectors[:0]
        filtered_reset = filtered.reset_index(drop=True)

        t0 = time.perf_counter()
        qvec = embedder.embed([q["question"]], is_query=True)[0]
        hits = cosine_search(qvec, filtered_vectors, top_k=max_k) if len(filtered_reset) else []
        latency = time.perf_counter() - t0

        top_similarity = hits[0][1] if hits else 0.0
        top_ocids_by_k = {}
        relevant = set(q.get("relevant_ocids", []))
        for k in k_values:
            retrieved = {filtered_reset.iloc[i]["ocid"] for i, _ in hits[:k]}
            top_ocids_by_k[k] = bool(retrieved & relevant) if relevant else None

        per_question.append({
            "id": q["id"], "category": q["category"], "n_candidates_after_filters": len(filtered),
            "top_similarity": top_similarity, "hit_at_k": top_ocids_by_k, "latency_seconds": round(latency, 4),
        })

    in_scope = [p for p in per_question if p["category"] == "in_scope"]
    out_scope = [p for p in per_question if p["category"] == "out_of_scope"]

    recall_at_k = {}
    for k in k_values:
        n_hits = sum(1 for p in in_scope if p["hit_at_k"].get(k))
        recall_at_k[k] = round(n_hits / len(in_scope), 3) if in_scope else None

    in_min = round(min(p["top_similarity"] for p in in_scope), 4) if in_scope else None
    out_max = round(max(p["top_similarity"] for p in out_scope), 4) if out_scope else None
    suggested_threshold = round((in_min + out_max) / 2, 3) if (in_min is not None and out_max is not None) else None

    results = {
        "vector_dimension": manifest["dimension"],
        "row_count": manifest["row_count"],
        "recall_at_k": recall_at_k,
        "mean_query_latency_seconds": round(statistics.mean(p["latency_seconds"] for p in per_question), 4),
        "in_scope_top1_similarity": {"min": in_min},
        "out_of_scope_top1_similarity": {"max": out_max},
        "threshold_calibration": {
            "in_scope_min": in_min, "out_of_scope_max": out_max,
            "suggested_threshold": suggested_threshold,
            "clean_separation": (in_min > out_max) if (in_min is not None and out_max is not None) else None,
        },
        "configured_threshold": cfg["rag_engine"]["similarity_threshold"],
        "per_question": per_question,
    }

    (eval_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(results, logs_dir / "phase4_evaluation_report.md")
    return results


def _write_markdown(results: dict, out_path) -> None:
    lines = ["# Fase 4 — Evaluación del RAG híbrido (Tarea 2)\n"]
    lines.append(f"Índice: {results['row_count']} procesos, dimensión {results['vector_dimension']}\n")
    lines.append("| k | Recall@k |")
    lines.append("|---|---|")
    for k, v in results["recall_at_k"].items():
        lines.append(f"| {k} | {v} |")
    lines.append(f"\nLatencia media de consulta: {results['mean_query_latency_seconds']}s\n")
    cal = results["threshold_calibration"]
    lines.append("## Calibración del umbral de abstención")
    lines.append(f"- Similitud top-1 mínima in-scope: **{cal['in_scope_min']}**")
    lines.append(f"- Similitud top-1 máxima out-of-scope: **{cal['out_of_scope_max']}**")
    lines.append(f"- Separación limpia: **{cal['clean_separation']}**")
    lines.append(f"- Umbral sugerido: **{cal['suggested_threshold']}**")
    lines.append(f"- Umbral configurado en config.yaml: **{results['configured_threshold']}**")
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    res = run_eval()
    print(f"Recall@k: {res['recall_at_k']}")
    print(f"Calibración: {res['threshold_calibration']}")
