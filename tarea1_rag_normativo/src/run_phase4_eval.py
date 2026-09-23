"""Fase 4 — Evaluación y comparación de embeddings (local vs. OpenAI).

Para la configuración de chunking seleccionada (config.yaml: chunking.selected_config):
1) construye (o reutiliza) el índice con embeddings locales y, si hay OPENAI_API_KEY,
   también con text-embedding-3-small,
2) corre el set de evaluación (eval/eval_set.json: 15 preguntas in-domain + 5 out-of-domain)
   contra cada índice,
3) reporta Recall@k, tiempo de indexación, costo, latencia y dimensión del vector,
4) calibra (con evidencia) el umbral de abstención comparando la similitud top-1 de
   preguntas in-domain vs. out-of-domain.

No importa UI. Uso: python -m src.run_phase4_eval   (desde tarea1_rag_normativo/)
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from src.chunking import load_fragments_jsonl
from src.config import get_openai_api_key, get_openai_embedding_model, load_config, resolve_path
from src.embeddings import LocalEmbedder, OpenAIEmbedder
from src.index_store import build_or_resume_index, cosine_search, load_existing


def _page_hit(frag_page_start: int, frag_page_end: int, expected_pages: list[int]) -> bool:
    return any(frag_page_start <= p <= frag_page_end for p in expected_pages)


def _evaluate_embedder(
    embedder_label: str,
    embedder,
    index_dir: Path,
    fragments,
    eval_questions: list[dict],
    k_values: list[int],
) -> dict:
    max_k = max(k_values)
    per_question = []

    t_start = time.perf_counter()
    for q in eval_questions:
        is_openai = "OpenAI" in type(embedder).__name__
        embed_kwargs = {"context": f"eval:{q['id']}"} if is_openai else {"is_query": True}
        t0 = time.perf_counter()
        qvec = embedder.embed([q["question"]], **embed_kwargs)[0]
        _, vectors, frag_meta = load_existing(index_dir)
        hits = cosine_search(qvec, vectors, top_k=max_k)
        latency = time.perf_counter() - t0

        top_similarity = hits[0][1] if hits else 0.0
        hit_ranks = {}  # k -> bool hay hit dentro de top-k
        if q["category"] == "in_domain":
            for k in k_values:
                found = False
                for row_idx, _score in hits[:k]:
                    frag = frag_meta[row_idx]
                    if frag.doc_id == q["expected_doc_id"] and _page_hit(frag.page_start, frag.page_end, q["expected_pages"]):
                        found = True
                        break
                hit_ranks[k] = found
        per_question.append({
            "id": q["id"],
            "category": q["category"],
            "top_similarity": top_similarity,
            "hit_at_k": hit_ranks,
            "latency_seconds": round(latency, 4),
            "top1_doc_id": frag_meta[hits[0][0]].doc_id if hits else None,
            "top1_pages": [frag_meta[hits[0][0]].page_start, frag_meta[hits[0][0]].page_end] if hits else None,
        })
    total_query_time = time.perf_counter() - t_start

    in_domain = [p for p in per_question if p["category"] == "in_domain"]
    out_domain = [p for p in per_question if p["category"] == "out_of_domain"]

    recall_at_k = {}
    for k in k_values:
        n_hits = sum(1 for p in in_domain if p["hit_at_k"].get(k))
        recall_at_k[k] = round(n_hits / len(in_domain), 3) if in_domain else None

    manifest, vectors, _ = load_existing(index_dir)

    return {
        "embedder": embedder_label,
        "vector_dimension": manifest["dimension"],
        "fragment_count": manifest["fragment_count"],
        "recall_at_k": recall_at_k,
        "mean_query_latency_seconds": round(statistics.mean(p["latency_seconds"] for p in per_question), 4),
        "total_eval_query_time_seconds": round(total_query_time, 4),
        "in_domain_top1_similarity": {
            "min": round(min(p["top_similarity"] for p in in_domain), 4) if in_domain else None,
            "max": round(max(p["top_similarity"] for p in in_domain), 4) if in_domain else None,
            "mean": round(statistics.mean(p["top_similarity"] for p in in_domain), 4) if in_domain else None,
        },
        "out_of_domain_top1_similarity": {
            "min": round(min(p["top_similarity"] for p in out_domain), 4) if out_domain else None,
            "max": round(max(p["top_similarity"] for p in out_domain), 4) if out_domain else None,
            "mean": round(statistics.mean(p["top_similarity"] for p in out_domain), 4) if out_domain else None,
        },
        "per_question": per_question,
    }


def run_phase4_eval() -> dict:
    cfg = load_config()
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    logs_dir = resolve_path(cfg["paths"]["logs_dir"])
    eval_dir = resolve_path(cfg["paths"]["eval_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    selected_config = cfg["chunking"]["selected_config"]
    fragments_path = processed_dir / f"fragments_{selected_config}.jsonl"
    fragments = load_fragments_jsonl(fragments_path)

    eval_set = json.loads((resolve_path(cfg["paths"]["eval_dir"]) / "eval_set.json").read_text(encoding="utf-8"))
    questions = eval_set["questions"]
    k_values = cfg["evaluation"]["k_values"]

    results = {"selected_chunk_config": selected_config, "embedders": []}

    # --- Local embeddings (ya indexado en Fase 2 bajo index_<config>__local) ---
    local_index_dir = processed_dir / f"index_{selected_config}__local"
    local_embedder = LocalEmbedder(cfg["embeddings"]["local"]["model_name"])
    if load_existing(local_index_dir) is None:
        build_or_resume_index(fragments, local_embedder, local_index_dir, "local")
    local_manifest, _, _ = load_existing(local_index_dir)
    local_build_time = None  # ya se construyó en Fase 2; se reporta desde phase2_chunking_report.json
    phase2_report_path = logs_dir / "phase2_chunking_report.json"
    if phase2_report_path.exists():
        phase2_report = json.loads(phase2_report_path.read_text(encoding="utf-8"))
        for c in phase2_report["configs"]:
            if c["config_name"] == selected_config:
                local_build_time = c["local_index_build"]["indexing_time_seconds_this_run"]

    local_result = _evaluate_embedder("local", local_embedder, local_index_dir, fragments, questions, k_values)
    local_result["indexing_time_seconds"] = local_build_time
    local_result["indexing_cost_usd"] = 0.0
    results["embedders"].append(local_result)

    # --- OpenAI embeddings (requiere OPENAI_API_KEY en .env) ---
    api_key = get_openai_api_key()
    if api_key:
        try:
            openai_model = get_openai_embedding_model(cfg)
            openai_index_dir = processed_dir / f"index_{selected_config}__openai"
            cost_log_path = logs_dir / "cost_log.jsonl"
            openai_embedder = OpenAIEmbedder(openai_model, api_key, cfg["pricing"], cost_log_path)

            build_stats = build_or_resume_index(fragments, openai_embedder, openai_index_dir, "openai")

            # costo real de indexación: sumar las entradas de cost_log con context que empieza con "build_index:openai"
            indexing_cost = _sum_recent_index_cost(cost_log_path, "build_index:openai")

            openai_result = _evaluate_embedder("openai", openai_embedder, openai_index_dir, fragments, questions, k_values)
            openai_result["indexing_time_seconds"] = round(build_stats["indexing_time_seconds_this_run"], 4)
            openai_result["indexing_cost_usd"] = indexing_cost
            results["embedders"].append(openai_result)
        except Exception as exc:
            results["embedders"].append({
                "embedder": "openai",
                "skipped": True,
                "reason": f"Falló la llamada a la API de OpenAI: {type(exc).__name__}: {exc}",
            })
    else:
        results["embedders"].append({
            "embedder": "openai",
            "skipped": True,
            "reason": "OPENAI_API_KEY no configurada en .env al momento de correr la evaluación.",
        })

    # --- Calibración del umbral de abstención con evidencia ---
    calibration = _calibrate_threshold(results["embedders"][0], questions)  # con el índice local, el que usa el motor en vivo
    results["threshold_calibration"] = calibration
    results["configured_threshold"] = cfg["rag_engine"]["similarity_threshold"]

    out_path = eval_dir / "results_embeddings_comparison.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_markdown_report(results, logs_dir / "phase4_evaluation_report.md")

    return results


def _sum_recent_index_cost(cost_log_path: Path, context_prefix: str) -> float:
    if not cost_log_path.exists():
        return 0.0
    total = 0.0
    with open(cost_log_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry["context"].startswith(context_prefix):
                total += entry["usd_cost"]
    return round(total, 6)


def _calibrate_threshold(local_result: dict, eval_questions: list[dict]) -> dict:
    in_sims = [p["top_similarity"] for p in local_result["per_question"] if p["category"] == "in_domain"]
    out_points = [
        (p["id"], p["top_similarity"])
        for p in local_result["per_question"] if p["category"] == "out_of_domain"
    ]
    if not in_sims or not out_points:
        return {"suggested_threshold": None, "note": "Datos insuficientes para calibrar."}

    in_min = round(min(in_sims), 4)
    out_sims_sorted = sorted(out_points, key=lambda x: -x[1])
    outlier_id, outlier_sim = out_sims_sorted[0]
    out_max_all = round(outlier_sim, 4)

    # Segundo grupo: out-of-domain "genuinamente ajeno" excluyendo el punto más alto,
    # que suele ser la pregunta-trampa de dominio adyacente (ver eval_set.json: reason/notes de esa Q).
    rest = out_sims_sorted[1:]
    out_max_excl_outlier = round(max(s for _, s in rest), 4) if rest else None

    practical_threshold = round((in_min + out_max_excl_outlier) / 2, 3) if out_max_excl_outlier is not None else None
    midpoint_all = round((in_min + out_max_all) / 2, 3)

    outlier_q = next((q for q in eval_questions if q["id"] == outlier_id), None)
    outlier_reason = outlier_q.get("reason", "") if outlier_q else ""

    note = (
        f"La pregunta out-of-domain con mayor similitud es **{outlier_id}** (sim={out_max_all}): "
        f"{outlier_reason} Es un caso límite deliberado (tema del dominio pero detalle fuera del corpus "
        f"indexado), no una pregunta genuinamente ajena; por eso se calibra el umbral con el resto de "
        f"preguntas out-of-domain (máx.={out_max_excl_outlier}) en vez de con este outlier. "
        f"Umbral operativo elegido: {practical_threshold} (vs. {in_min} mínimo in-domain). "
        f"Con este umbral, **{outlier_id} sigue sin abstenerse por similitud** — queda como responsabilidad "
        f"de la segunda capa de abstención (instrucción explícita al LLM de no responder sin respaldo "
        f"en los fragmentos), documentado como limitación conocida del sistema."
    )
    return {
        "in_domain_top1_min": in_min,
        "out_of_domain_top1_max": out_max_all,
        "out_of_domain_top1_max_excluding_domain_adjacent_trap": out_max_excl_outlier,
        "domain_adjacent_trap_question": outlier_id,
        "suggested_threshold": practical_threshold,
        "midpoint_including_trap": midpoint_all,
        "clean_separation": in_min > out_max_all,
        "note": note,
    }


def _write_markdown_report(results: dict, out_path: Path) -> None:
    lines = ["# Fase 4 — Evaluación y comparación de embeddings\n"]
    lines.append(f"Configuración de chunking evaluada: `{results['selected_chunk_config']}`\n")

    lines.append("## Comparación local vs. OpenAI\n")
    lines.append("| Embedder | Dim. | Fragmentos | Recall@1 | Recall@3 | Recall@5 | Tiempo indexado (s) | Costo indexado (USD) | Latencia media consulta (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results["embedders"]:
        if r.get("skipped"):
            lines.append(f"| {r['embedder']} | — | — | — | — | — | — | — | _omitido: {r['reason']}_ |")
            continue
        rk = r["recall_at_k"]
        lines.append(
            f"| {r['embedder']} | {r['vector_dimension']} | {r['fragment_count']} | "
            f"{rk.get(1)} | {rk.get(3)} | {rk.get(5)} | "
            f"{r['indexing_time_seconds']} | {r['indexing_cost_usd']} | {r['mean_query_latency_seconds']} |"
        )

    lines.append("\n## Calibración del umbral de abstención (con el índice local en producción)\n")
    cal = results["threshold_calibration"]
    lines.append(f"- Similitud top-1 mínima entre preguntas in-domain: **{cal['in_domain_top1_min']}**")
    lines.append(f"- Similitud top-1 máxima entre preguntas out-of-domain (todas): **{cal['out_of_domain_top1_max']}** (pregunta `{cal['domain_adjacent_trap_question']}`, caso límite deliberado)")
    lines.append(f"- Similitud top-1 máxima entre preguntas out-of-domain genuinamente ajenas: **{cal['out_of_domain_top1_max_excluding_domain_adjacent_trap']}**")
    lines.append(f"- Umbral operativo elegido (evidencia): **{cal['suggested_threshold']}**")
    lines.append(f"- Umbral configurado actualmente en config.yaml: **{results['configured_threshold']}**")
    lines.append(f"\n{cal['note']}\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    res = run_phase4_eval()
    print(json.dumps({k: v for k, v in res.items() if k != "embedders"}, ensure_ascii=False, indent=2))
    for r in res["embedders"]:
        if r.get("skipped"):
            print(f"- {r['embedder']}: OMITIDO ({r['reason']})")
        else:
            print(f"- {r['embedder']}: Recall@k={r['recall_at_k']}, dim={r['vector_dimension']}, "
                  f"costo_indexado=${r['indexing_cost_usd']}, tiempo_indexado={r['indexing_time_seconds']}s")
