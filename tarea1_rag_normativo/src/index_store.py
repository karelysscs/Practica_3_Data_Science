"""Fase 2 — Construcción y carga de índices vectoriales.

Formato simple y sin dependencias pesadas de base de datos: los vectores se
guardan en un .npy y la metadata de cada fragmento en un .jsonl paralelo
(mismo orden de filas). Un manifest.json permite que la construcción del
índice sea idempotente y resumible: si el índice ya contiene un fragment_id,
no se vuelve a embeber (evita recomputar y, en el caso de OpenAI, evita costo
duplicado).

No importa UI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.chunking import Fragment


def _manifest_path(index_dir: Path) -> Path:
    return index_dir / "manifest.json"


def _vectors_path(index_dir: Path) -> Path:
    return index_dir / "vectors.npy"


def _fragments_path(index_dir: Path) -> Path:
    return index_dir / "fragments.jsonl"


def load_existing(index_dir: Path) -> tuple[dict, np.ndarray, list[Fragment]] | None:
    manifest_p, vectors_p, fragments_p = _manifest_path(index_dir), _vectors_path(index_dir), _fragments_path(index_dir)
    if not (manifest_p.exists() and vectors_p.exists() and fragments_p.exists()):
        return None
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    vectors = np.load(vectors_p)
    fragments = []
    with open(fragments_p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                fragments.append(Fragment(**json.loads(line)))
    return manifest, vectors, fragments


def build_or_resume_index(
    fragments: list[Fragment],
    embedder,
    index_dir: Path,
    embedder_label: str,
    embed_batch_size: int = 32,
) -> dict:
    """Construye el índice desde cero o reanuda uno parcial, embebiendo solo los
    fragment_ids que falten. Devuelve un dict con estadísticas del build
    (tiempo total de indexación, cuántos fragmentos ya estaban, cuántos se
    embebieron en esta corrida)."""
    import time

    index_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing(index_dir)

    existing_ids: set[str] = set()
    existing_vectors = None
    existing_fragments: list[Fragment] = []
    if existing is not None:
        _, existing_vectors, existing_fragments = existing
        existing_ids = {f.fragment_id for f in existing_fragments}

    missing = [f for f in fragments if f.fragment_id not in existing_ids]

    t0 = time.perf_counter()
    if missing:
        texts = [f.text for f in missing]
        kwargs = {"batch_size": embed_batch_size} if "OpenAI" not in type(embedder).__name__ else {"batch_size": embed_batch_size, "context": f"build_index:{embedder_label}"}
        new_vectors = embedder.embed(texts, **kwargs)
    else:
        new_vectors = np.zeros((0, existing_vectors.shape[1] if existing_vectors is not None else embedder.dimension), dtype="float32")
    elapsed = time.perf_counter() - t0

    if existing_vectors is not None and existing_vectors.shape[0] > 0:
        all_vectors = np.vstack([existing_vectors, new_vectors]) if new_vectors.shape[0] else existing_vectors
    else:
        all_vectors = new_vectors

    all_fragments = existing_fragments + missing

    np.save(_vectors_path(index_dir), all_vectors)
    with open(_fragments_path(index_dir), "w", encoding="utf-8") as f:
        for frag in all_fragments:
            f.write(json.dumps(frag.__dict__, ensure_ascii=False) + "\n")

    manifest = {
        "embedder_label": embedder_label,
        "dimension": int(all_vectors.shape[1]) if all_vectors.shape[0] else None,
        "fragment_count": len(all_fragments),
        "fragment_ids": [f.fragment_id for f in all_fragments],
    }
    _manifest_path(index_dir).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "index_dir": str(index_dir),
        "total_fragments": len(all_fragments),
        "already_indexed": len(existing_ids),
        "newly_indexed": len(missing),
        "indexing_time_seconds_this_run": round(elapsed, 4),
        "vector_dimension": manifest["dimension"],
    }


def cosine_search(query_vector: np.ndarray, matrix: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Devuelve [(row_index, score)] ordenado descendente. Asume filas ya
    normalizadas L2 (los embedders de este proyecto normalizan al indexar)."""
    q = query_vector / (np.linalg.norm(query_vector) + 1e-12)
    scores = matrix @ q
    top_k = min(top_k, len(scores))
    idx = np.argpartition(-scores, top_k - 1)[:top_k] if top_k < len(scores) else np.arange(len(scores))
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]
