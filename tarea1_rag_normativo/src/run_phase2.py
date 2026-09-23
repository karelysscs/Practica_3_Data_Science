"""Orquestador de la Fase 2: para cada configuración de chunking en config.yaml —
1) genera fragmentos de ambos documentos con esa configuración,
2) escribe fragments_<config>.jsonl,
3) construye el índice local (embeddings gratis, sin costo) para esa config,
4) reporta conteo de fragmentos y distribución de longitudes.

La configuración seleccionada (config.yaml: chunking.selected_config) es la que
usa el motor RAG (Fase 3) y la comparación de embeddings (Fase 4).

Idempotente: build_or_resume_index no recomputa fragmentos ya embebidos.
Uso: python -m src.run_phase2   (ejecutado desde tarea1_rag_normativo/)
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from src.chunking import chunk_document, write_fragments_jsonl
from src.config import load_config, resolve_path
from src.embeddings import LocalEmbedder
from src.index_store import build_or_resume_index


def run_phase2() -> dict:
    cfg = load_config()
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    logs_dir = resolve_path(cfg["paths"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    local_embedder = LocalEmbedder(cfg["embeddings"]["local"]["model_name"])

    report = {"configs": []}

    for chunk_cfg in cfg["chunking"]["configs"]:
        config_name = chunk_cfg["name"]
        all_fragments = []
        for doc_cfg in cfg["documents"]:
            clean_path = processed_dir / f"{doc_cfg['id']}_pages_clean.jsonl"
            fragments = chunk_document(
                doc_cfg=doc_cfg,
                clean_jsonl_path=clean_path,
                chunk_size_chars=chunk_cfg["chunk_size_chars"],
                chunk_overlap_chars=chunk_cfg["chunk_overlap_chars"],
                chunk_config_name=config_name,
            )
            all_fragments.extend(fragments)

        frag_out_path = processed_dir / f"fragments_{config_name}.jsonl"
        write_fragments_jsonl(all_fragments, frag_out_path)

        lengths = [f.char_count for f in all_fragments]
        per_doc_counts = {}
        for f in all_fragments:
            per_doc_counts[f.doc_id] = per_doc_counts.get(f.doc_id, 0) + 1

        # Verificación de unicidad de IDs (requisito de Fase 2)
        ids = [f.fragment_id for f in all_fragments]
        n_unique = len(set(ids))

        # Índice local para esta configuración (rápido, sin costo -> se construye
        # para las dos configs, y sirve de base para elegir selected_config)
        index_dir = processed_dir / f"index_{config_name}__local"
        build_stats = build_or_resume_index(
            fragments=all_fragments,
            embedder=local_embedder,
            index_dir=index_dir,
            embedder_label="local",
            embed_batch_size=32,
        )

        config_report = {
            "config_name": config_name,
            "chunk_size_chars": chunk_cfg["chunk_size_chars"],
            "chunk_overlap_chars": chunk_cfg["chunk_overlap_chars"],
            "total_fragments": len(all_fragments),
            "unique_fragment_ids": n_unique,
            "ids_are_unique": n_unique == len(ids),
            "fragments_per_document": per_doc_counts,
            "length_chars_min": min(lengths) if lengths else 0,
            "length_chars_max": max(lengths) if lengths else 0,
            "length_chars_mean": round(statistics.mean(lengths), 1) if lengths else 0,
            "length_chars_median": statistics.median(lengths) if lengths else 0,
            "length_chars_stdev": round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0,
            "fragments_file": str(frag_out_path.relative_to(resolve_path("."))),
            "local_index_build": build_stats,
        }
        report["configs"].append(config_report)

    report["selected_config"] = cfg["chunking"]["selected_config"]

    report_path = logs_dir / "phase2_chunking_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_markdown_summary(report, logs_dir / "phase2_chunking_report.md")

    return report


def _write_markdown_summary(report: dict, out_path: Path) -> None:
    lines = ["# Fase 2 — Comparación de configuraciones de chunking\n"]
    lines.append("| Config | tamaño | overlap | fragmentos | IDs únicos | min | mediana | media | max | stdev |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in report["configs"]:
        lines.append(
            f"| {c['config_name']} | {c['chunk_size_chars']} | {c['chunk_overlap_chars']} | "
            f"{c['total_fragments']} | {'✅' if c['ids_are_unique'] else '❌'} | "
            f"{c['length_chars_min']} | {c['length_chars_median']} | {c['length_chars_mean']} | "
            f"{c['length_chars_max']} | {c['length_chars_stdev']} |"
        )
    lines.append(f"\n**Configuración seleccionada para Fases 3-5:** `{report['selected_config']}`\n")
    lines.append(
        "\nCriterio de selección: ambas configuraciones se indexaron y se corrieron contra las "
        "20 preguntas de `eval/eval_set.json` (Fase 4). Con el modelo de embeddings local "
        "(`intfloat/multilingual-e5-small`), fragmentos más CORTOS (config_a, 800/120) dieron "
        "mejor Recall@5 (0.733) que fragmentos más largos (config_b, 1500/250: Recall@5=0.6): "
        "fragmentos más pequeños aíslan mejor un artículo/numeral específico, lo que ayuda cuando "
        "la pregunta (sobre todo las coloquiales) apunta a un dato puntual dentro de un artículo "
        "largo. Ver el detalle completo, incluida la calibración del umbral de abstención, en "
        "`logs/phase4_evaluation_report.md` y `eval/results_embeddings_comparison.json`."
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = run_phase2()
    print(json.dumps({k: v for k, v in result.items() if k != "configs"}, ensure_ascii=False, indent=2))
    for c in result["configs"]:
        print(f"- {c['config_name']}: {c['total_fragments']} fragmentos, únicos={c['ids_are_unique']}, "
              f"long. media={c['length_chars_mean']}")
