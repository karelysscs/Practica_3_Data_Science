"""Fase 3 — Índice vectorial sobre las DESCRIPCIONES de los procesos (no sobre
condiciones numéricas/territoriales: esas se aplican como filtros estructurados
en hybrid_rag.py, tal como pide el issue). No importa UI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_embedding_text(row: pd.Series) -> str:
    parts = [
        row.get("tender_title") or "",
        row.get("tender_description") or "",
        f"Comprador: {row.get('buyer_name') or ''}",
        f"Departamento: {row.get('buyer_department') or ''}",
        f"Categoría: {row.get('main_category') or ''}",
        f"Modalidad: {row.get('procurement_method_details') or ''}",
    ]
    return ". ".join(p for p in parts if p.strip())


def build_index(df: pd.DataFrame, embedder, index_dir: Path, chunk_size: int = 500, progress: bool = True) -> dict:
    """Construye el índice por LOTES, guardando progreso parcial en disco tras
    cada lote (vectors.partial.npy + .progress). Si el proceso se interrumpe
    (Ctrl+C, corte de energía, kill), volver a llamar a esta función RETOMA
    desde el último lote guardado en vez de re-embeber todo desde cero —
    la misma idea de "construcción de índice idempotente y resumible" que la
    Fase 2 de la Tarea 1, aplicada aquí porque son ~18k textos y el embebido
    puede tardar minutos en una laptop bajo carga."""
    index_dir.mkdir(parents=True, exist_ok=True)
    texts = df.apply(build_embedding_text, axis=1).tolist()
    n = len(texts)

    partial_path = index_dir / "vectors.partial.npy"
    progress_path = index_dir / ".progress"

    done = 0
    vectors_chunks = []
    if partial_path.exists() and progress_path.exists():
        done = int(progress_path.read_text(encoding="utf-8").strip())
        existing = np.load(partial_path)
        if existing.shape[0] == done:
            vectors_chunks.append(existing)
        else:
            done = 0  # inconsistente: se reinicia por seguridad

    while done < n:
        end = min(done + chunk_size, n)
        batch_vectors = embedder.embed(texts[done:end], is_query=False)
        vectors_chunks.append(batch_vectors)
        done = end

        combined_so_far = np.vstack(vectors_chunks)
        np.save(partial_path, combined_so_far)
        progress_path.write_text(str(done), encoding="utf-8")
        vectors_chunks = [combined_so_far]  # evita reconcatenar desde cero cada vez

        if progress:
            print(f"  [index] {done}/{n} fragmentos embebidos ({100*done/n:.1f}%)", flush=True)

    vectors = vectors_chunks[0]
    np.save(index_dir / "vectors.npy", vectors)
    df[["ocid"]].to_json(index_dir / "ocid_order.json", orient="records")
    manifest = {"dimension": int(vectors.shape[1]), "row_count": len(df), "embedder_model": embedder.model_name}
    (index_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    partial_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    return manifest


def load_index(index_dir: Path) -> tuple[np.ndarray, dict]:
    vectors = np.load(index_dir / "vectors.npy")
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    return vectors, manifest
